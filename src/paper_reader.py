"""
Paper upload handling: save PDFs under data/papers, extract text and embedded images (PyMuPDF) for the paper-reading UI.
"""

from pathlib import Path
import hashlib
import fitz  # PyMuPDF
from src.paths import DATA_DIR


PAPER_DIR = DATA_DIR / "papers"
IMAGE_DIR = DATA_DIR / "paper_images"


def ensure_dirs():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_pdf(uploaded_file) -> Path:
    ensure_dirs()
    safe_name = Path(uploaded_file.name.replace('\\', '/')).name
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file.read()
    if not raw.startswith(b'%PDF-') or len(raw) > 50 * 1024 * 1024:
        raise ValueError("pdf_invalid_or_too_large")
    file_id = hashlib.sha256(raw).hexdigest()
    pdf_path = PAPER_DIR / f"{file_id}_{safe_name}"
    if pdf_path.exists() and hashlib.sha256(pdf_path.read_bytes()).hexdigest() == file_id:
        return pdf_path

    with open(pdf_path, "wb") as f:
        f.write(raw)
    pdf_path.chmod(0o600)

    return pdf_path


def extract_paper_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    texts = []

    for page in doc:
        # 先用最直接的文本提取
        texts.append(page.get_text("text"))

    doc.close()
    return "\n".join(texts).strip()


def extract_paper_markdown(pdf_path: str) -> str:
    """
    可选：如果你装了 pymupdf4llm，就用它做更适合 LLM 的提取。
    没装也没关系，可以先不用这个函数。
    """
    import pymupdf4llm
    return pymupdf4llm.to_markdown(pdf_path)


def extract_images_from_pdf(pdf_path: str, max_images: int = 12) -> list[dict]:
    ensure_dirs()
    doc = fitz.open(pdf_path)
    results = []
    seen_xrefs = set()

    for page_index in range(doc.page_count):
        image_list = doc.get_page_images(page_index)

        for img in image_list:
            xref = img[0]

            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                pix = fitz.Pixmap(doc, xref)

                # 转 RGB，避免某些颜色空间不能直接保存
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                image_name = f"{Path(pdf_path).stem}_p{page_index+1}_{xref}.png"
                image_path = IMAGE_DIR / image_name
                pix.save(str(image_path))

                results.append({
                    "page": page_index + 1,
                    "xref": xref,
                    "path": str(image_path),
                    "name": image_name
                })

                if len(results) >= max_images:
                    doc.close()
                    return results

            except Exception:
                continue

    doc.close()
    return results
