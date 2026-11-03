"""Structure-aware PDF evidence using optional Docling and the existing PyMuPDF.

All extracted source/page/figure labels originate in document text. Coordinates
are top-left PDF points. Crops are rendered only for selected evidence.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import re
import statistics
import copy
import threading
from collections import OrderedDict
import fitz
from src.paths import DATA_DIR
from src.splitter import chunk_text
from src.vector_store import chunk_id, tokenize

_NUMBER = r'(?:\d+[a-z]?|[IVXLC]+)'
CAPTION = re.compile(r'^\s*((?:fig(?:ure)?\.?|table|图|表|abb(?:ildung)?\.?|tabelle|tableau)\s*'+_NUMBER+r')(?![A-Za-z0-9])', re.I)
REFERENCE = re.compile(r'(fig(?:ure)?s?\.?|tables?|图|表|abb\.?|abbildungen?|tabellen?|tableaux?)\s*'
                       r'('+_NUMBER+r'(?:\s*(?:,|and|und|et|和|与|&)\s*'+_NUMBER+r')*)(?![A-Za-z0-9])', re.I)


def reference_keys(text):
    keys = set()
    for match in REFERENCE.finditer(text):
        kind = _kind(match.group(1))
        for number in re.findall(r'(?<![A-Za-z0-9])'+_NUMBER+r'(?![A-Za-z0-9])', match.group(2), re.I):
            keys.add((kind, number.casefold()))
    return keys


def _kind(label):
    return 'table' if re.match(r'(table|表|tabelle|tableau)', label, re.I) else 'figure'


def _base(pdf_path, source, page, section, kind, text, bbox=None, label=None):
    return {'source': source, 'page': page, 'section': section, 'kind': kind,
            'text': text, 'bbox': list(bbox) if bbox else None,
            'label': label, 'pdf_path': str(Path(pdf_path).resolve())}


def _pymupdf_blocks(pdf_path, source):
    blocks, warnings = [], []
    section = ''
    with fitz.open(pdf_path) as pdf:
        for number, page in enumerate(pdf, 1):
            layout = page.get_text('dict')
            sizes = [round(s['size'], 1) for b in layout['blocks'] if b['type'] == 0
                     for line in b['lines'] for s in line['spans'] if s['text'].strip()]
            body = statistics.median(sizes) if sizes else 10
            raw = []
            page_start_section = section
            images = [b for b in layout['blocks'] if b['type'] == 1]
            captions = []
            for block in layout['blocks']:
                if block['type'] != 0:
                    continue
                text = '\n'.join(''.join(s['text'] for s in line['spans']) for line in block['lines']).strip()
                if not text:
                    continue
                match = CAPTION.match(text)
                maximum = max((s['size'] for line in block['lines'] for s in line['spans']), default=body)
                if match:
                    label = match.group(1)
                    captions.append((label, text, block['bbox']))
                    kind = 'caption'
                elif len(text) < 160 and (maximum > body*1.15 or re.match(r'^\d+(?:\.\d+)*\s+[A-Z]', text)):
                    kind = 'heading'
                else:
                    kind = 'paragraph'
                raw.append((block['bbox'], text, kind))
            for bbox, text, kind in raw:
                if kind == 'heading':
                    section = text
                blocks.append(_base(pdf_path, source, number, section, kind, text, bbox))
            def section_at(box):
                current = page_start_section
                for raw_box, raw_text, raw_kind in raw:
                    if raw_kind == 'heading' and raw_box[1] <= box[1]:
                        current = raw_text
                return current
            for label, text, bbox in captions:
                kind = _kind(label)
                # Prefer an actual embedded figure; otherwise a selected full-page
                # crop preserves vector diagrams and tables without guessing bounds.
                candidates = [b for b in images if b['bbox'][3] <= bbox[1]+30]
                closest = min(candidates, key=lambda b: abs(b['bbox'][3]-bbox[1]), default=None)
                crop = closest['bbox'] if closest and kind == 'figure' else list(page.rect)
                blocks.append({**_base(pdf_path, source, number, section_at(bbox), kind, text, crop, label),
                               'visual_only': True})
            labelled_boxes = [tuple(b['bbox']) for b in blocks if b['page'] == number and b['kind'] == 'figure']
            for image in images:
                if tuple(image['bbox']) not in labelled_boxes:
                    blocks.append(_base(pdf_path, source, number, section_at(image['bbox']), 'figure',
                                        f'Embedded image on page {number}; no confirmed caption.', image['bbox']))
            if hasattr(page, 'find_tables'):
                try:
                    for table in page.find_tables().tables:
                        rows = table.extract()
                        text = '\n'.join(' | '.join(str(v or '') for v in row) for row in rows)
                        nearby = [(lab, cap, box) for lab, cap, box in captions if _kind(lab) == 'table'
                                  and abs(box[3]-table.bbox[1]) < 160]
                        label, caption, _ = min(nearby, key=lambda x: abs(x[2][3]-table.bbox[1]),
                                                default=(None, '', None))
                        if label:
                            blocks = [b for b in blocks if not (b.get('page') == number and
                                      b.get('kind') == 'table' and b.get('label') == label and b.get('visual_only'))]
                        blocks.append(_base(pdf_path, source, number, section_at(table.bbox), 'table',
                                            (caption+'\n'+text).strip(), table.bbox, label))
                except Exception:
                    warnings.append('table_extraction_partial')
            if not raw:
                # Scans remain retrievable by document/page; no fake OCR text.
                blocks.append(_base(pdf_path, source, number, section, 'page',
                                    f'Page {number}: no extractable text. Visual inspection required.', list(page.rect)))
    return blocks, sorted(set(warnings))


def _docling_blocks(pdf_path, source):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import PictureItem, TableItem
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    # Crops use the existing PyMuPDF renderer rather than retaining all page bitmaps.
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    document = converter.convert(str(pdf_path)).document
    blocks, section = [], ''
    with fitz.open(pdf_path) as pdf:
        for item, _level in document.iterate_items():
            prov = getattr(item, 'prov', None)
            if not prov:
                continue
            page = int(prov[0].page_no)
            if not 1 <= page <= len(pdf):
                continue
            box = prov[0].bbox.to_top_left_origin(page_height=pdf[page-1].rect.height)
            bbox = [box.l, box.t, box.r, box.b]
            item_label = getattr(getattr(item, 'label', ''), 'value', str(getattr(item, 'label', '')))
            text = getattr(item, 'text', '') or ''
            label = None
            kind = 'paragraph'
            if isinstance(item, (PictureItem, TableItem)):
                caption = item.caption_text(document) or ''
                match = CAPTION.match(caption)
                label = match.group(1) if match else None
                if isinstance(item, TableItem):
                    kind = 'table'
                    text = caption + '\n' + item.export_to_dataframe(doc=document).to_csv(index=False)
                else:
                    kind, text = 'figure', caption or f'Unlabelled figure on page {page}.'
            elif item_label in ('section_header', 'title'):
                kind, section = 'heading', text
            elif item_label == 'caption':
                kind = 'caption'
            if text.strip():
                blocks.append(_base(pdf_path, source, page, section, kind, text.strip(), bbox, label))
    if not blocks:
        raise ValueError('Docling returned no evidence')
    return blocks


def structure_chunks(blocks, chunk_size=800, overlap=150):
    # Never join across a document/page/section boundary. Long blocks use the
    # original splitter as a fallback rather than discarding their metadata.
    if chunk_size <= overlap or overlap < 0:
        raise ValueError('Require chunk_size > overlap >= 0')
    chunks, pending = [], None
    def flush():
        nonlocal pending
        if pending:
            pending['chunk_id'] = chunk_id(pending)
            chunks.append(pending)
            pending = None
    for block in blocks:
        visual = block['kind'] in ('figure', 'table', 'page')
        if visual:
            flush()
        for text in chunk_text(block['text'], chunk_size, overlap):
            current = {**block, 'text': text}
            same = pending and all(pending.get(k) == current.get(k) for k in ('source', 'page', 'section', 'kind'))
            if not visual and same and len(pending['text'])+len(text)+2 <= chunk_size:
                pending['text'] += '\n\n'+text
            else:
                flush()
                pending = current
            if visual:
                flush()
    flush()
    return chunks


_PARSE_CACHE = OrderedDict()
_PARSE_LOCK = threading.RLock()


def parse_paper(pdf_path, source=None, *, parser='pymupdf', chunk_size=800, overlap=150):
    pdf_path = Path(pdf_path)
    if pdf_path.stat().st_size > 50*1024*1024:
        raise ValueError('pdf_size')
    with fitz.open(pdf_path) as pdf:
        if pdf.is_encrypted or len(pdf) > 500:
            raise ValueError('pdf_encrypted_or_too_long')
    source = source or pdf_path.name
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    cache_key = (str(pdf_path.resolve()), digest, source, parser, chunk_size, overlap, 2)
    with _PARSE_LOCK:
        if cache_key in _PARSE_CACHE:
            _PARSE_CACHE.move_to_end(cache_key)
            return copy.deepcopy(_PARSE_CACHE[cache_key])
    warnings, used = [], 'pymupdf'
    if parser == 'docling':
        try:
            blocks = _docling_blocks(pdf_path, source)
            used = 'docling'
        except Exception:
            blocks, warnings = _pymupdf_blocks(pdf_path, source)
            warnings.append('docling_fallback')
    else:
        blocks, warnings = _pymupdf_blocks(pdf_path, source)
    chunks = structure_chunks(blocks, chunk_size, overlap)
    for chunk in chunks:
        chunk['document_sha256'] = digest
    visuals = [c for c in chunks if c['kind'] in ('figure', 'table', 'page')]
    result = {'source': source,
            'text': '\n\n'.join(f"[{source}, p. {b['page']}]\n{b['text']}"
                                for b in blocks if b['kind'] not in ('figure', 'page')),
            'chunks': chunks, 'visuals': visuals, 'parser': used, 'warnings': warnings}
    # Failed advanced parsing is retried on the next explicit import attempt.
    if 'docling_fallback' not in warnings:
        with _PARSE_LOCK:
            _PARSE_CACHE[cache_key] = copy.deepcopy(result)
            while len(_PARSE_CACHE) > 8:
                _PARSE_CACHE.popitem(last=False)
    return result


def visual_question(query):
    return bool(re.search(r'fig(?:ure)?|table|chart|diagram|architecture|ablation|visual|图|表|架构|消融|abbildung|tabelle|diagramm|tableau|schéma|architecture', query, re.I))


def render_evidence_image(evidence):
    path = Path(evidence['pdf_path']).resolve()
    if (DATA_DIR / 'papers').resolve() not in path.parents:
        raise ValueError('pdf_path')
    content_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if evidence.get('document_sha256') and content_digest != evidence['document_sha256']:
        raise ValueError('pdf_changed_reparse')
    digest = hashlib.sha256((content_digest+str(evidence['page'])+str(evidence.get('bbox'))).encode()).hexdigest()[:24]
    output = DATA_DIR / 'paper_images' / f'{digest}.png'
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open(path) as pdf:
            page = pdf[int(evidence['page'])-1]
            clip = fitz.Rect(evidence.get('bbox') or page.rect) & page.rect
            if clip.is_empty:
                clip = page.rect
            scale = min(2.0, 2000/max(clip.width, clip.height))
            page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False).save(output)
        output.chmod(0o600)
    return {'path': str(output.relative_to(DATA_DIR)), 'name': evidence.get('label') or f"page {evidence['page']}",
            **{key:evidence[key] for key in ('source','page','label','chunk_id') if evidence.get(key) is not None}}


def select_visual_evidence(query, retrieved, all_chunks, max_visuals=3, *, force=False):
    if max_visuals <= 0 or (not force and not visual_question(query)
                            and not any(d.get('visual_only') for d in retrieved)):
        return [], []
    max_visuals = min(4, max_visuals)
    refs = reference_keys(query)
    numbers = set(re.findall(r'(?:page|seite|第)\s*(\d+)', query, re.I))
    pages = {(d.get('source'), d.get('page')) for d in retrieved}
    terms = set(tokenize(query))
    ranked = []
    for doc in all_chunks:
        if doc.get('kind') not in ('figure', 'table', 'page') or not doc.get('pdf_path'):
            continue
        exact = bool(reference_keys(doc.get('label') or '') & refs)
        overlap = len(terms & set(tokenize(doc['text'])))
        matched_page = (doc.get('source'), doc.get('page')) in pages
        page_requested = str(doc.get('page')) in numbers
        if exact or overlap or matched_page or page_requested:
            ranked.append((100*exact+40*page_requested+5*matched_page+overlap, doc))
    ranked.sort(key=lambda x: -x[0])
    if refs:
        # An explicit figure/table number must not pad the request with unrelated logos.
        ranked = [pair for pair in ranked if reference_keys(pair[1].get('label') or '') & refs]
    selected, images, seen = [], [], set()
    for _score, doc in ranked:
        # Deduplicate full-page crops and split table chunks.
        key = (doc.get('pdf_path'), doc.get('page'), str(doc.get('bbox')))
        if key in seen:
            if not any(d.get('chunk_id') == doc.get('chunk_id') for d in selected):
                selected.append(doc)
            continue
        if len(images) >= max_visuals:
            continue
        image = render_evidence_image(doc)
        selected.append(doc)
        images.append(image)
        seen.add(key)
    return selected, images
