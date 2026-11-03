"""Validated local attachments. Base64 exists only while building API requests."""
from __future__ import annotations
import base64
import hashlib
import io
from pathlib import Path
import warnings
from collections import OrderedDict
from PIL import Image, ImageOps, UnidentifiedImageError
from src.paths import DATA_DIR

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_IMAGES = 4
MAX_SIDE = 2000
FORMATS = {"JPEG", "PNG", "GIF", "WEBP"}
IMAGE_DIR = DATA_DIR / "uploads" / "images"
_IMAGE_CACHE = OrderedDict()


class MediaError(ValueError):
    pass


def save_image(raw: bytes, name: str = "image") -> dict:
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise MediaError("image_size")
    cache_key = (hashlib.sha256(raw).hexdigest(),name,str(IMAGE_DIR.resolve()))
    cached = _IMAGE_CACHE.get(cache_key)
    if cached and (DATA_DIR/cached['path']).is_file():
        _IMAGE_CACHE.move_to_end(cache_key)
        return cached.copy()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as probe:
                if probe.format not in FORMATS:
                    raise MediaError("image_format")
                if probe.width * probe.height > MAX_PIXELS:
                    raise MediaError("image_pixels")
                probe.verify()
            with Image.open(io.BytesIO(raw)) as original:
                frame = ImageOps.exif_transpose(original).convert("RGB")
                frame.thumbnail((MAX_SIDE, MAX_SIDE))
                buffer = io.BytesIO()
                frame.save(buffer, format="JPEG", quality=90)
                content = buffer.getvalue()
                width, height = frame.size
    except MediaError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning,
            Image.DecompressionBombError):
        raise MediaError("image_format") from None
    digest = hashlib.sha256(content).hexdigest()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_DIR / f"{digest}.jpg"
    if not path.exists():
        path.write_bytes(content)
        path.chmod(0o600)
    result = {"id": digest, "name": Path(name.replace('\\', '/')).name[:120],
            "path": str(path.relative_to(DATA_DIR)), "mime": "image/jpeg",
            "width": width, "height": height}
    _IMAGE_CACHE[cache_key] = result.copy()
    while len(_IMAGE_CACHE) > 64:
        _IMAGE_CACHE.popitem(last=False)
    return result


def attachment_path(attachment: dict) -> Path:
    candidate = Path(attachment.get("path", ""))
    path = (candidate if candidate.is_absolute() else DATA_DIR / candidate).resolve()
    allowed = [(DATA_DIR / "uploads" / "images").resolve(),
               (DATA_DIR / "paper_images").resolve()]
    if not any(root in path.parents for root in allowed):
        raise MediaError("image_path")
    if not path.is_file():
        raise MediaError("image_missing")
    return path


def image_block(attachment: dict) -> dict:
    path = attachment_path(attachment)
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise MediaError("image_size")
    with Image.open(path) as img:
        if img.format not in FORMATS or img.width * img.height > MAX_PIXELS:
            raise MediaError("image_format")
        mime = Image.MIME[img.format]
    content = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{content}"}}


def unique_attachments(attachments):
    seen, result = set(), []
    for attachment in attachments:
        identity = attachment.get('id') or attachment.get('path')
        if identity not in seen:
            result.append(attachment)
            seen.add(identity)
    return result


def user_content(text: str, attachments=None):
    attachments = attachments or []
    if len(attachments) > MAX_IMAGES:
        raise MediaError("image_count")
    attachments = unique_attachments(attachments)
    if not attachments:
        return text
    blocks = [{'type':'text','text':text}]
    for attachment in attachments:
        if attachment.get('source'):
            from src.citations import source_label
            citation = f"[{attachment['citation']}] " if attachment.get('citation') else ''
            blocks.append({'type':'text','text':citation+'Image evidence: '+source_label(attachment)})
        blocks.append(image_block(attachment))
    return blocks


def bind_image_citations(images, docs):
    ids = {d.get('chunk_id'):f'S{i}' for i,d in enumerate(docs,1) if d.get('chunk_id')}
    return [{**image, **({'citation':ids[image['chunk_id']]}
                          if image.get('chunk_id') in ids else {})} for image in images or []]


def chat_messages(system: str, history: list[dict], query: str, attachments=None):
    attachments = attachments or []
    if len(attachments) > MAX_IMAGES:
        raise MediaError("image_count")
    attachments = unique_attachments(attachments)
    remaining = MAX_IMAGES - len(attachments)
    seen = {a.get('id') or a.get('path') for a in attachments}
    chosen = {}
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user":
            images = []
            for attachment in history[i].get('attachments') or []:
                identity = attachment.get('id') or attachment.get('path')
                if remaining > len(images) and identity not in seen:
                    images.append(attachment)
                    seen.add(identity)
            chosen[i] = images
            remaining -= len(images)
    messages = [{"role": "system", "content": system}]
    for i, msg in enumerate(history):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = str(msg.get("content", ""))
        content = user_content(text, chosen.get(i)) if role == "user" else text
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content(query, attachments)})
    return messages


def contains_images(messages):
    return any(isinstance(m.get("content"), list) and
               any(b.get("type") == "image_url" for b in m["content"]) for m in messages)
