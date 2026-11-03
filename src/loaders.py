"""
Extract plain text from user-uploaded documents: PDF (pypdf) and DOCX (python-docx), selected by file extension.
"""

from pypdf import PdfReader
from docx import Document


def load_document_chunks(file, *, chunk_size=800, overlap=150, parser='pymupdf'):
    """Metadata-preserving ingestion; original extract_text remains available."""
    from src.paper_reader import save_uploaded_pdf
    from src.paper_structure import parse_paper, structure_chunks
    from src.splitter import chunk_text
    from src.vector_store import chunk_id
    if file.name.lower().endswith('.pdf'):
        path = save_uploaded_pdf(file)
        return parse_paper(path, file.name, parser=parser, chunk_size=chunk_size, overlap=overlap)['chunks']
    if file.name.lower().endswith('.docx'):
        document = Document(file)
        blocks, section = [], ''
        for paragraph in document.paragraphs:
            if not paragraph.text.strip():
                continue
            heading = paragraph.style and paragraph.style.name.startswith('Heading')
            if heading:
                section = paragraph.text
            blocks.append({'source': file.name, 'page': None, 'section': section,
                           'kind': 'heading' if heading else 'paragraph', 'text': paragraph.text})
        for table in document.tables:
            blocks.append({'source': file.name, 'page': None, 'section': '', 'kind': 'table',
                           'text': '\n'.join(' | '.join(cell.text for cell in row.cells) for row in table.rows)})
        return structure_chunks(blocks, chunk_size, overlap)
    docs = [{'source': file.name, 'text': text, 'kind': 'paragraph'}
            for text in chunk_text(extract_text(file), chunk_size, overlap)]
    for doc in docs:
        doc['chunk_id'] = chunk_id(doc)
    return docs


def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def extract_text_from_docx(file) -> str:
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text(file) -> str:
    suffix = file.name.lower().split(".")[-1]

    if suffix == "pdf":
        return extract_text_from_pdf(file)
    elif suffix == "docx":
        return extract_text_from_docx(file)
    elif suffix == "txt":
        return file.read().decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")
