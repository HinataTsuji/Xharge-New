"""Image and overlay helpers."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
BGR_DARK_GRAY = (30, 30, 30)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode user-uploaded image bytes into BGR numpy image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes")
    return image


def encode_png(image_bgr: np.ndarray) -> bytes:
    """Encode BGR numpy image into PNG bytes."""
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode PNG")
    return encoded.tobytes()


def save_png(image_bgr: np.ndarray, output_path: Path) -> None:
    """Persist BGR image to disk as PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encode_png(image_bgr))


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def png_bytes_to_base64(data: bytes) -> str:
    """Convert PNG bytes to base64 data URI fragment."""
    return base64.b64encode(data).decode("utf-8")


def blank_canvas(width: int, height: int, color: Tuple[int, int, int] = BGR_DARK_GRAY) -> np.ndarray:
    """Create blank BGR canvas."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = color
    return canvas


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR array."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array to PIL image."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def image_dimensions(image_bgr: np.ndarray) -> Tuple[int, int]:
    """Return image width and height."""
    h, w = image_bgr.shape[:2]
    return w, h


def png_size_bytes(image_bgr: np.ndarray) -> int:
    """Return encoded PNG byte size without saving to disk."""
    return len(encode_png(image_bgr))


def read_image(path: Path) -> np.ndarray:
    """Load image from filesystem as BGR."""
    data = path.read_bytes()
    return decode_image_bytes(data)


def write_image(path: Path, image_bgr: np.ndarray) -> None:
    """Write image via PIL for compatibility with many environments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = bgr_to_pil(image_bgr)
    buf = BytesIO()
    image.save(buf, format="PNG")
    path.write_bytes(buf.getvalue())
