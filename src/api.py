from __future__ import annotations

import base64
import hashlib
import io
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Literal, Optional

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from PIL import Image

from analysis import analyse_arrays
from app_core import (
    DISPLAY_SIZE,
    MODEL_SIZE,
    REPORTLAB_OK,
    SafeCaeID,
    apply_strokes,
    bytesToRgb,
    draw_grid,
    ensureRgbSize,
    ensure_mask01_size,
    fileFingerprint,
    make_pdf_report,
    overlay_mask,
    postprocess_keras_output,
    preprocess_for_keras,
)
from db import delete_case, init_db, load_app_session, load_case_visits, save_app_session, save_visit
from llm_report import generate_report_with_llm
from ulcer_unet_infer import load_ulcer_unet, predict_mask_from_path

init_db()

APP_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from tensorflow import keras

    KERAS_OK = True
    KERAS_ERROR = None
except Exception as e1:
    try:
        import keras  # type: ignore

        KERAS_OK = True
        KERAS_ERROR = None
    except Exception as e2:
        KERAS_OK = False
        KERAS_ERROR = f"tf.keras: {e1} | keras: {e2}"

app = FastAPI(title="Corneal Ulcer Assessment API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:5174").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024)

SessionMode = Literal["Fluorescein", "white"]


class SessionData(dict):
    pass


FLUOR_MODEL: Any = None
WHITE_MODEL: tuple[Any, Any] | None = None


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    try:
        request_size = int(content_length) if content_length else 0
    except ValueError:
        request_size = 0
    if request_size > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."},
        )
    return await call_next(request)


def get_session(session_id: str | None) -> tuple[str, SessionData]:
    sid = session_id or str(uuid.uuid4())
    stored = load_app_session(sid)
    if stored is None:
        state = SessionData(
            case_id="",
            visit_date="",
            mode=None,
            session_eye="Right",
        )
        save_app_session(sid, state)
        return sid, state
    return sid, SessionData(stored)


def persist_session(session_id: str, state: SessionData) -> None:
    save_app_session(session_id, state)


async def read_upload_bytes(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def np_to_png_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def data_url_to_np(data_url: str) -> np.ndarray:
    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    raw = base64.b64decode(payload)
    return np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))


def mask_to_data_url(mask01: np.ndarray) -> str:
    return np_to_png_data_url(mask01.astype(np.uint8) * 255)


def get_fluor_model(model_path: str):
    global FLUOR_MODEL
    if not KERAS_OK:
        raise RuntimeError(f"Install TensorFlow/Keras to use Fluorescein model. Import error: {KERAS_ERROR}")
    if FLUOR_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found:{model_path}")
        if hasattr(keras, "saving") and hasattr(keras.saving, "load_model"):
            FLUOR_MODEL = keras.saving.load_model(model_path, compile=False)  # type: ignore[attr-defined]
        else:
            FLUOR_MODEL = keras.models.load_model(model_path, compile=False)  # type: ignore[attr-defined]
    return FLUOR_MODEL


def get_white_model(ckpt_path: str):
    global WHITE_MODEL
    if WHITE_MODEL is None:
        WHITE_MODEL = load_ulcer_unet(ckpt_path)
    return WHITE_MODEL


def set_step(state: SessionData, ns: str, step: int) -> None:
    state[f"{ns}_step"] = int(step)


def step_state(state: SessionData, ns: str) -> int:
    return int(state.get(f"{ns}_step", 1))


def reset_editor_for_new_image(state: SessionData, ns: str, fp: str) -> None:
    last_fp_key = f"{ns}_last_fp"
    if state.get(last_fp_key) != fp:
        state[last_fp_key] = fp
        for k in [
            f"{ns}_mask01_display",
            f"{ns}_mask_confirmed",
            f"{ns}_mm_per_px",
            f"{ns}_ref_roi_xywh",
            f"{ns}_ref_rect_canvas",
            f"{ns}_linecanvas",
        ]:
            state.pop(k, None)
        set_step(state, ns, 2)


def session_payload(sid: str, state: SessionData, ns: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": sid,
        "case_id": state.get("case_id", ""),
        "visit_date": state.get("visit_date", ""),
        "mode": state.get("mode"),
        "session_eye": state.get("session_eye", "Right"),
    }
    if ns:
        payload["step"] = step_state(state, ns)
        payload["mm_per_px"] = state.get(f"{ns}_mm_per_px")
        payload["mask_confirmed"] = f"{ns}_mask_confirmed" in state
    return payload


class SessionUpdate(BaseModel):
    session_id: Optional[str] = None
    case_id: str = ""
    visit_date: str = ""
    session_eye: Literal["Right", "Left"] = "Right"


class ModeOpen(BaseModel):
    session_id: Optional[str] = None
    mode: SessionMode


class BrushApply(BaseModel):
    session_id: str
    ns: SessionMode
    mode: Literal["Add", "Erase"]
    canvas_rgba: str


class ConfirmMask(BaseModel):
    session_id: str
    ns: SessionMode


class CalibrationRequest(BaseModel):
    session_id: str
    ns: SessionMode
    method: Literal["Line", "Grid (research only)"]
    known_mm: Optional[float] = None
    line: Optional[dict[str, float]] = None
    grid_mm: Optional[float] = None
    spacing_px: Optional[int] = None


class GreyReference(BaseModel):
    use_ref: bool = False
    roi: Optional[list[int]] = None
    target_grey: float = 120.0


class ResultsRequest(BaseModel):
    session_id: str
    ns: SessionMode
    acquisition_notes: str = ""
    grey_reference: GreyReference = GreyReference()


class SaveVisitRequest(ResultsRequest):
    report_text: str


@app.get("/api/session")
def create_session():
    sid, state = get_session(None)
    return session_payload(sid, state)


@app.get("/api/health")
def health_check():
    init_db()
    return {"ok": True, "database": "ready"}


@app.post("/api/session")
def update_session(req: SessionUpdate):
    sid, state = get_session(req.session_id)
    state["case_id"] = req.case_id
    state["visit_date"] = req.visit_date
    state["session_eye"] = req.session_eye
    persist_session(sid, state)
    return session_payload(sid, state, "white" if state.get("mode") == "white" else state.get("mode"))


@app.post("/api/open")
def open_mode(req: ModeOpen):
    sid, state = get_session(req.session_id)
    state["mode"] = req.mode
    state.setdefault(f"{req.mode}_step", 1)
    persist_session(sid, state)
    return session_payload(sid, state, req.mode)


@app.post("/api/predict")
async def predict(
    session_id: str = Form(...),
    ns: SessionMode = Form(...),
    threshold: float = Form(0.5),
    image: UploadFile | None = File(None),
):
    sid, state = get_session(session_id)
    if image is not None:
        img_bytes = await read_upload_bytes(image)
        img_name = image.filename or "image"
        state[f"{ns}_last_name"] = img_name
        state[f"{ns}_last_bytes"] = img_bytes
    else:
        img_name = state.get(f"{ns}_last_name")
        img_bytes = state.get(f"{ns}_last_bytes")
        if not img_name or not isinstance(img_bytes, (bytes, bytearray)):
            raise HTTPException(status_code=400, detail=f"No {ns} image uploaded yet.")

    fp = fileFingerprint(str(img_name), bytes(img_bytes))
    reset_editor_for_new_image(state, ns, fp)

    if ns == "Fluorescein":
        model_path = os.environ.get("Fluorescein_MODEL_PATH", os.path.join(APP_DIR, "RealDataModelv2.keras"))
        rgb_native = bytesToRgb(bytes(img_bytes))
        rgb_display = ensureRgbSize(rgb_native, DISPLAY_SIZE)
        rgb_model = ensureRgbSize(rgb_native, MODEL_SIZE)
        pred_key = f"Fluorescein|{fp}|thr={threshold:.3f}|model={model_path}"
        if state.get("Fluorescein_pred_key") != pred_key:
            model = get_fluor_model(model_path)
            pred = model.predict(preprocess_for_keras(rgb_model), verbose=0)
            state["Fluorescein_pred_mask01_model"] = postprocess_keras_output(pred, threshold)
            state["Fluorescein_pred_key"] = pred_key
        pred_mask = ensure_mask01_size(state["Fluorescein_pred_mask01_model"], DISPLAY_SIZE)
    else:
        ckpt = os.environ.get("WHITE_CKPT_PATH", os.path.join(APP_DIR, "best.pt"))
        suffix = os.path.splitext(str(img_name))[1].lower()
        if suffix not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            suffix = ".png"
        pred_key = f"white|{fp}|thr={threshold:.3f}|ckpt={ckpt}"
        if state.get("white_pred_key") != pred_key:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(bytes(img_bytes))
                tmp_path = tmp.name
            try:
                torch_model, torch_device = get_white_model(ckpt)
                rgb_display, pred_mask = predict_mask_from_path(torch_model, torch_device, tmp_path, thr=threshold)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            state["white_pred_rgb_512"] = rgb_display
            state["white_pred_mask01_512"] = pred_mask
            state["white_pred_key"] = pred_key
        rgb_display = ensureRgbSize(state["white_pred_rgb_512"], DISPLAY_SIZE)
        pred_mask = ensure_mask01_size(state["white_pred_mask01_512"], DISPLAY_SIZE)

    state[f"{ns}_rgb_display"] = rgb_display
    state[f"{ns}_pred_mask01_display"] = pred_mask
    state.setdefault(f"{ns}_mask01_display", pred_mask.copy())
    if step_state(state, ns) < 2:
        set_step(state, ns, 2)

    edited_mask = state[f"{ns}_mask01_display"]
    persist_session(sid, state)
    return {
        **session_payload(sid, state, ns),
        "rgb": np_to_png_data_url(rgb_display),
        "pred_mask": mask_to_data_url(pred_mask),
        "edited_mask": mask_to_data_url(edited_mask),
        "overlay": np_to_png_data_url(overlay_mask(rgb_display, edited_mask, alpha=0.35)),
    }


@app.post("/api/mask/apply")
def apply_mask(req: BrushApply):
    sid, state = get_session(req.session_id)
    mask = state.get(f"{req.ns}_mask01_display")
    if mask is None:
        raise HTTPException(status_code=400, detail="No editable mask is available.")
    rgba = data_url_to_np(req.canvas_rgba)
    state[f"{req.ns}_mask01_display"] = apply_strokes(rgba, mask, req.mode)
    rgb = state[f"{req.ns}_rgb_display"]
    edited = state[f"{req.ns}_mask01_display"]
    persist_session(sid, state)
    return {
        **session_payload(sid, state, req.ns),
        "edited_mask": mask_to_data_url(edited),
        "overlay": np_to_png_data_url(overlay_mask(rgb, edited, alpha=0.35)),
    }


@app.post("/api/mask/reset")
def reset_mask(req: ConfirmMask):
    sid, state = get_session(req.session_id)
    pred = state.get(f"{req.ns}_pred_mask01_display")
    rgb = state.get(f"{req.ns}_rgb_display")
    if pred is None or rgb is None:
        raise HTTPException(status_code=400, detail="No prediction is available.")
    state[f"{req.ns}_mask01_display"] = pred.copy()
    persist_session(sid, state)
    return {
        **session_payload(sid, state, req.ns),
        "edited_mask": mask_to_data_url(pred),
        "overlay": np_to_png_data_url(overlay_mask(rgb, pred, alpha=0.35)),
    }


@app.post("/api/mask/confirm")
def confirm_mask(req: ConfirmMask):
    sid, state = get_session(req.session_id)
    mask = state.get(f"{req.ns}_mask01_display")
    if mask is None:
        raise HTTPException(status_code=400, detail="No editable mask is available.")
    state[f"{req.ns}_mask_confirmed"] = mask.copy()
    set_step(state, req.ns, 4)
    persist_session(sid, state)
    return session_payload(sid, state, req.ns)


@app.get("/api/calibration/base")
def calibration_base(session_id: str, ns: SessionMode, spacing_px: int = 25, grid_opacity: float = 0.16):
    _, state = get_session(session_id)
    rgb = state.get(f"{ns}_rgb_display")
    mask = state.get(f"{ns}_mask_confirmed", state.get(f"{ns}_mask01_display"))
    if rgb is None or mask is None:
        raise HTTPException(status_code=400, detail="No image/mask is available.")
    base = overlay_mask(rgb, mask, alpha=0.35)
    return {
        "base": np_to_png_data_url(base),
        "grid": np_to_png_data_url(draw_grid(base, spacing_px, thickness=1, opacity=grid_opacity)),
    }


@app.get("/api/overlay")
def current_overlay(session_id: str, ns: SessionMode, alpha: float = 0.35):
    _, state = get_session(session_id)
    rgb = state.get(f"{ns}_rgb_display")
    mask = state.get(f"{ns}_mask01_display")
    if rgb is None or mask is None:
        raise HTTPException(status_code=400, detail="No image/mask is available.")
    return {"overlay": np_to_png_data_url(overlay_mask(rgb, mask, alpha=float(alpha)))}


@app.post("/api/calibration")
def set_calibration(req: CalibrationRequest):
    sid, state = get_session(req.session_id)
    mm = None
    if req.method == "Line":
        if not req.known_mm or not req.line:
            raise HTTPException(status_code=400, detail="Line calibration requires known_mm and line.")
        x1, y1, x2, y2 = req.line["x1"], req.line["y1"], req.line["x2"], req.line["y2"]
        px_dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        if px_dist > 0:
            mm = float(req.known_mm) / px_dist
    else:
        if not req.grid_mm or not req.spacing_px:
            raise HTTPException(status_code=400, detail="Grid calibration requires grid_mm and spacing_px.")
        mm = float(req.grid_mm) / float(req.spacing_px)
    if not (isinstance(mm, float) and mm > 0):
        raise HTTPException(status_code=400, detail="Set a valid calibration first.")
    state[f"{req.ns}_mm_per_px"] = mm
    set_step(state, req.ns, 5)
    persist_session(sid, state)
    return session_payload(sid, state, req.ns)


def compute_results_payload(state: SessionData, ns: SessionMode, acq_notes: str, grey: GreyReference) -> dict[str, Any]:
    mm_per_px = state.get(f"{ns}_mm_per_px")
    if not (isinstance(mm_per_px, (int, float)) and float(mm_per_px) > 0):
        raise HTTPException(status_code=400, detail="Calibration not set (Go back to Step 4)")
    rgb = state.get(f"{ns}_rgb_display")
    mask = state.get(f"{ns}_mask_confirmed", state.get(f"{ns}_mask01_display"))
    if rgb is None or mask is None:
        raise HTTPException(status_code=400, detail="No confirmed mask is available.")

    eye = state.get("session_eye", "Right")
    case_id = SafeCaeID(state.get("case_id", ""))
    visit_date = (state.get("visit_date") or "").strip() or datetime.now().strftime("%Y-%m-%d")
    ref_roi = tuple(grey.roi) if ns == "white" and grey.use_ref and grey.roi is not None else None
    ref_target = float(grey.target_grey)

    result = analyse_arrays(
        rgb=rgb,
        mask01=mask,
        case_id=case_id if case_id else None,
        visit_date=visit_date,
        mm_per_pixel=float(mm_per_px),
        source="verified",
        compute_opacity=(ns == "white"),
        eye=eye,
        reference_roi_xywh=ref_roi,
        reference_target_grey=ref_target,
    )
    flags = result.get("analysis_flags") or []
    b = state.get(f"{ns}_last_bytes")
    nm = state.get(f"{ns}_last_name", "image")
    img_hash = fileFingerprint(nm, b) if isinstance(b, (bytes, bytearray)) else "unknown"
    summary = {
        "case_id": case_id if case_id else "—",
        "visit_date": visit_date,
        "eye": eye,
        "mode": ns,
        "area_mm2": None if result.get("area_mm2") is None else float(result.get("area_mm2")),
        "eq_diameter_mm": None if result.get("eq_diameter_mm") is None else float(result.get("eq_diameter_mm")),
        "zone": result.get("zone"),
        "vertical_sector": result.get("vertical_sector"),
        "horizontal_sector": result.get("horizontal_sector"),
        "opacity_zscore": None if result.get("opacity_zscore") is None else float(result.get("opacity_zscore")),
        "blur": float(result.get("blur_laplacian_var", 0.0)),
        "qc_flags": ";".join(flags),
        "mm_per_px": float(mm_per_px),
        "image_hash": img_hash,
    }
    llm_result = generate_report_with_llm(summary, acquisition_notes=acq_notes)
    report_text = llm_result.report_text
    if acq_notes.strip():
        report_text = report_text.rstrip() + "\n\nClinician notes\n" + acq_notes.strip()

    return {
        "result": result,
        "summary": summary,
        "flags": flags,
        "report_text": report_text,
        "report_source": "LLM" if llm_result.used_llm else "Template",
        "report_model": llm_result.used_model,
        "overlay": np_to_png_data_url(overlay_mask(rgb, mask, alpha=0.35)),
    }


@app.post("/api/results")
def results(req: ResultsRequest):
    _, state = get_session(req.session_id)
    payload = compute_results_payload(state, req.ns, req.acquisition_notes, req.grey_reference)
    case_id = SafeCaeID(state.get("case_id", ""))
    visits = load_case_visits(case_id) if case_id else []
    rows = [
        {
            "visit_date": v.visit_date,
            "eye": v.eye,
            "mode": v.mode,
            "area_mm2": v.area_mm2,
            "eq_diameter_mm": v.eq_diameter_mm,
            "opacity_zscore": v.opacity_zscore,
            "mm_per_px": v.mm_per_px,
            "qc_flags": v.qc_flags,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in visits
    ]
    return {**payload, "timeline": rows}


@app.post("/api/save")
def save(req: SaveVisitRequest):
    _, state = get_session(req.session_id)
    payload = compute_results_payload(state, req.ns, req.acquisition_notes, req.grey_reference)
    summary = payload["summary"]
    case_id = SafeCaeID(state.get("case_id", ""))
    if not case_id:
        raise HTTPException(status_code=400, detail="Add a Case ID to save this visit to the timeline.")
    visit_id = hashlib.sha1(
        f"{case_id}|{summary['visit_date']}|{summary['eye']}|{req.ns}|{summary['image_hash']}".encode("utf-8")
    ).hexdigest()[:16]
    try:
        save_visit(
            visit_id=visit_id,
            case_id=case_id,
            visit_date=summary["visit_date"],
            eye=summary["eye"],
            mode=req.ns,
            image_hash=summary["image_hash"],
            mm_per_px=float(summary["mm_per_px"]),
            analysis_result=payload["result"],
            report_text=req.report_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/api/delete-case")
def delete_case_endpoint(req: SessionUpdate):
    sid, state = get_session(req.session_id)
    case_id = SafeCaeID(req.case_id or state.get("case_id", ""))
    if not case_id:
        raise HTTPException(status_code=400, detail="Enter a Case ID first.")
    delete_case(case_id)
    return {"ok": True, **session_payload(sid, state)}


@app.post("/api/report")
def report(req: ResultsRequest):
    _, state = get_session(req.session_id)
    payload = compute_results_payload(state, req.ns, req.acquisition_notes, req.grey_reference)
    summary = payload["summary"]
    rgb = state[f"{req.ns}_rgb_display"]
    mask = state.get(f"{req.ns}_mask_confirmed", state[f"{req.ns}_mask01_display"])
    if REPORTLAB_OK:
        try:
            data = make_pdf_report(
                SafeCaeID(state.get("case_id", "")) or "CASE",
                summary["visit_date"],
                summary["eye"],
                summary,
                rgb,
                mask,
                payload["report_text"],
            )
            return Response(
                data,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": (
                        f"attachment; filename={(SafeCaeID(state.get('case_id', '')) or 'CASE')}_"
                        f"{summary['visit_date']}_{summary['eye']}_report.pdf"
                    )
                },
            )
        except Exception:
            pass
    return Response(
        payload["report_text"].encode("utf-8"),
        media_type="text/plain",
        headers={
            "Content-Disposition": (
                f"attachment; filename={(SafeCaeID(state.get('case_id', '')) or 'CASE')}_"
                f"{summary['visit_date']}_{summary['eye']}_report.txt"
            )
        },
    )
