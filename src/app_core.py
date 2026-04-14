from __future__ import annotations

import hashlib
import json
import os
import uuid
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas as rl_canvas

    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

DISPLAY_SIZE = 512
MODEL_SIZE = 256
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data_store")
DATASET_DIR = os.path.join(DATA_DIR, "anon_dataset")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)


def ensureRgbSize(rgb: np.ndarray, size: int) -> np.ndarray:
    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb


def ensure_mask01_size(mask01: np.ndarray, size: int) -> np.ndarray:
    if mask01.shape[0] != size or mask01.shape[1] != size:
        mask01 = cv2.resize(mask01.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask01 > 0).astype(np.uint8)


def overlay_mask(rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    base = rgb.copy()
    over = base.copy()
    over[mask01 == 1] = np.array([255, 0, 0], dtype=np.uint8)
    return cv2.addWeighted(over, float(alpha), base, 1 - float(alpha), 0)


def draw_grid(rgb: np.ndarray, spacing_px: int, thickness: int = 1, opacity: float = 0.2) -> np.ndarray:
    h, w = rgb.shape[:2]
    grid = rgb.copy()
    color = (255, 255, 255)
    spacing_px = max(1, int(spacing_px))
    for x in range(0, w, spacing_px):
        cv2.line(grid, (x, 0), (x, h), color, int(thickness))
    for y in range(0, h, spacing_px):
        cv2.line(grid, (0, y), (w, y), color, int(thickness))
    return cv2.addWeighted(grid, float(opacity), rgb, 1 - float(opacity), 0)


def fileFingerprint(name: str, b: bytes) -> str:
    h = hashlib.sha256()
    h.update(name.encode("utf-8", errors="ignore"))
    h.update(b)
    return h.hexdigest()[:16]


def bytesToRgb(b: bytes) -> np.ndarray:
    img = Image.open(BytesIO(b)).convert("RGB")
    return np.array(img)


def SafeCaeID(case_id: str) -> str:
    case_id = (case_id or "").strip()
    return case_id if case_id else ""


def SaveAnonymisedSample(rgb: np.ndarray, mask01: np.ndarray, meta: dict) -> str:
    sample_id = str(uuid.uuid4())
    sample_dir = os.path.join(DATASET_DIR, sample_id)
    os.makedirs(sample_dir, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(sample_dir, "image.png"))
    Image.fromarray((mask01.astype(np.uint8) * 255)).save(os.path.join(sample_dir, "mask.png"))
    with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return sample_id


def explainability_overlay(rgb_u8: np.ndarray, mask01: np.ndarray, alpha_mask: float = 0.35) -> np.ndarray:
    return overlay_mask(rgb_u8, mask01, alpha=alpha_mask)


def preprocess_for_keras(rgb_u8: np.ndarray) -> np.ndarray:
    x = rgb_u8.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)


def postprocess_keras_output(pred: np.ndarray, threshold: float) -> np.ndarray:
    p = np.squeeze(np.array(pred))
    if p.ndim == 3:
        p = p[..., 0]
    if p.min() < -0.01 or p.max() > 1.01:
        p = 1.0 / (1.0 + np.exp(-p))
    mask01 = (p >= float(threshold)).astype(np.uint8)
    return ensure_mask01_size(mask01, MODEL_SIZE)


def apply_strokes(canvas_rgba: np.ndarray, mask01_display: np.ndarray, mode: str) -> np.ndarray:
    if canvas_rgba is None:
        return mask01_display
    rgba = canvas_rgba.astype(np.uint8)
    if rgba.shape[-1] == 4:
        a = rgba[..., 3]
        painted = a > 0
        new_mask = mask01_display.copy()
        if mode == "Add":
            new_mask[painted] = 1
        else:
            new_mask[painted] = 0
        return ensure_mask01_size(new_mask, DISPLAY_SIZE)
    r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    if mode == "Add":
        painted = (g > 150) & (r < 170) & (b < 170)
    else:
        painted = (r > 160) & (b > 160) & (g < 180)
    new_mask = mask01_display.copy()
    if mode == "Add":
        new_mask[painted] = 1
    else:
        new_mask[painted] = 0
    return ensure_mask01_size(new_mask, DISPLAY_SIZE)


def make_pdf_report(
    case_id: str,
    visit_date: str,
    eye: str,
    summary: dict,
    rgb: np.ndarray,
    mask: np.ndarray,
    report_text: str,
) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab not installed")
    fig = explainability_overlay(rgb, mask, alpha_mask=0.35)
    pil = Image.fromarray(fig)
    bio = BytesIO()
    pil.save(bio, format="PNG")
    bio.seek(0)

    buf = BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 50, "Corneal ulcer assessment (assistive)")

    c.setFont("Helvetica", 11)
    c.drawString(40, h - 75, f"Case: {case_id}")
    c.drawString(40, h - 92, f"Date: {visit_date}")
    c.drawString(40, h - 109, f"Eye: {eye}")
    c.drawString(40, h - 126, f"Modality: {summary.get('mode', '')}")

    y = h - 160
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Key metrics")
    c.setFont("Helvetica", 11)
    y -= 18

    def keyMetricsHelper(label, val):
        nonlocal y
        c.drawString(50, y, f"{label}: {'' if val is None else val}")
        y -= 16

    keyMetricsHelper("Area (mm²)", summary.get("area_mm2"))
    keyMetricsHelper("Equivalent diameter (mm)", summary.get("eq_diameter_mm"))
    keyMetricsHelper("Zone", summary.get("zone"))
    keyMetricsHelper("Vertical", summary.get("vertical_sector"))
    keyMetricsHelper("Horizontal", summary.get("horizontal_sector"))
    keyMetricsHelper("Opacity z-score", summary.get("opacity_zscore"))
    keyMetricsHelper("Blur", summary.get("blur"))
    keyMetricsHelper("Calibration (mm/px)", summary.get("mm_per_px"))
    keyMetricsHelper("QC flags", summary.get("qc_flags"))

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y - 10, "Overlay")
    img = ImageReader(bio)
    img_w = 320
    img_h = 320
    img_y = y - 24 - img_h
    if img_y < 120:
        c.showPage()
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, h - 50, "Corneal ulcer assessment (assistive)")
        c.setFont("Helvetica", 11)
        c.drawString(40, h - 75, f"Case: {case_id}")
        c.drawString(40, h - 92, f"Date: {visit_date}")
        c.drawString(40, h - 109, f"Eye: {eye}")
        c.drawString(40, h - 126, f"Modality: {summary.get('mode', '')}")
        y = h - 160
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y - 10, "Overlay")
        img_y = y - 24 - img_h
    c.drawImage(img, 40, img_y, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, img_y - 18, "Report")
    c.setFont("Helvetica", 9)

    def wrapLine(line: str, max_w: float):
        words = (line or "").split()
        if not words:
            return [""]
        out = []
        cur = ""
        for w0 in words:
            test = w0 if cur == "" else cur + " " + w0
            if stringWidth(test, "Helvetica", 9) <= max_w:
                cur = test
            else:
                if cur:
                    out.append(cur)
                cur = w0
        if cur:
            out.append(cur)
        return out

    text_obj = c.beginText(40, img_y - 34)
    bottom = 50
    max_w = w - 80
    for raw in report_text.splitlines():
        for line in wrapLine(raw, max_w):
            if text_obj.getY() < bottom:
                c.drawText(text_obj)
                c.showPage()
                c.setFont("Helvetica", 9)
                text_obj = c.beginText(40, h - 175)
            text_obj.textLine(line)
    c.drawText(text_obj)
    c.save()
    buf.seek(0)
    return buf.read()
