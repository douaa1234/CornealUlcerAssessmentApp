#Corneal Ulcer Assessment streamlit app.py
import os
import inspect
import hashlib
import tempfile
import json
import uuid
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import cv2
import streamlit as st
import pandas as pd
from PIL import Image

from analysis import analyse_arrays
from ulcer_unet_infer import load_ulcer_unet,predict_mask_from_path
from llm_report import generate_report_with_llm
from db import init_db, save_visit, load_case_visits

# Initialising dataset
init_db()

#To ccheck if these are installed and enable features accordingly 
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader

    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

#fluerscin images (keras model )
try:
    from tensorflow import keras
    KERAS_OK = True
except Exception:
    try:
        import keras
        KERAS_OK = True
    except Exception:
        KERAS_OK = False

#Brush editor 
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except Exception:
    CANVAS_OK = False

DISPLAY_SIZE = 512
MODEL_SIZE = 256

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data_store")
DATASET_DIR = os.path.join(DATA_DIR, "anon_dataset")
os.makedirs(DATA_DIR,exist_ok=True)
os.makedirs(DATASET_DIR,exist_ok=True)


# UI helpers
#this is to fix the compatiobility issue with othe versions of streamlit 
#see wether st.image(img, **img_kw()) supports use_container_width
def img_kw():
    sig = inspect.signature(st.image)
    if "use_container_width" in sig.parameters:
        return {"use_container_width": True}
    return {"use_column_width": True}

#Style fo the page 
def inject_css():
    st.markdown(
        """
        <style>
        @media (prefers-color-scheme: dark) {
          :root{
            --bd: rgba(255,255,255,0.10);
            --muted: rgba(255,255,255,0.75);
            --shadow2: 0 6px 18px rgba(0,0,0,0.22);
            --panel-bg: rgba(255,255,255,0.03);
            --appbar-bg: rgba(255,255,255,0.025);
            --text: rgba(255,255,255,0.92);
            --text-soft: rgba(255,255,255,0.78);
            --r: 16px;
          }
        }
        @media (prefers-color-scheme: light) {
          :root{
            --bd: rgba(0,0,0,0.10);
            --muted: rgba(0,0,0,0.62);
            --shadow2: 0 6px 18px rgba(0,0,0,0.10);
            --panel-bg: rgba(0,0,0,0.02);
            --appbar-bg: rgba(0,0,0,0.015);
            --text: rgba(0,0,0,0.90);
            --text-soft: rgba(0,0,0,0.70);
            --r: 16px;
          }
        }

        .block-container{ max-width: 1180px; padding-top: 1.25rem; padding-bottom: 2rem; }
        .appbar{
          display:flex; align-items:center; justify-content:space-between;
          padding:12px 14px; border-radius: var(--r);
          border:1px solid var(--bd);
          background: var(--appbar-bg);
          box-shadow: var(--shadow2); margin-bottom: 14px;
        }
        .brand{ font-size:1.05rem; font-weight:800; letter-spacing:-0.02em; color: var(--text); }
        .subtitle{ font-size:0.86rem; color: var(--muted); margin-top: 2px; }
        .panel{
          border: 1px solid var(--bd);
          border-radius: var(--r);
          background: var(--panel-bg);
          box-shadow: var(--shadow2);
          padding: 12px 12px;
          margin-bottom: 12px;
        }
        .panelTitle{ font-weight: 780; letter-spacing:-0.02em; margin-bottom: 6px; color: var(--text); }
        .stepper{ display:flex; flex-wrap:wrap; gap: 8px; margin: 8px 0 4px 0; }
        .chip{
          display:flex; align-items:center; gap: 8px;
          padding: 7px 10px;
          border-radius: 999px;
          border: 1px solid var(--bd);
          background: var(--panel-bg);
          font-size: 0.82rem;
          color: var(--text-soft);
        }
        .chip b{ color: var(--text); }
        .dot{
          width: 10px; height: 10px; border-radius: 999px;
          background: rgba(127,127,127,0.35);
          border: 1px solid var(--bd);
        }
        .dot.on{ background: rgba(46, 204, 113, 0.90); border-color: rgba(46, 204, 113, 0.55); }
        .sticky{ position: sticky; top: 12px; }

        /* Hide header anchor icons I got this bit of code with the help of Chatgpt */
        [data-testid="stMarkdownContainer"] h1 a,
        [data-testid="stMarkdownContainer"] h2 a,
        [data-testid="stMarkdownContainer"] h3 a,
        [data-testid="stMarkdownContainer"] h4 a,
        [data-testid="stMarkdownContainer"] h5 a,
        [data-testid="stMarkdownContainer"] h6 a { display: none !important; }
        a[aria-label="Anchor link"]{ display:none !important; }
        a.anchor-link, a.stMarkdownAnchorLink{ display:none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

#Image/mask utilities
#enforces square image size 
def ensureRgbSize(rgb: np.ndarray, size: int) -> np.ndarray:
    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb

#enforceS square binary mask size
def ensure_mask01_size(mask01: np.ndarray, size: int) -> np.ndarray:
    if mask01.shape[0] != size or mask01.shape[1] != size:
        mask01 = cv2.resize(mask01.astype(np.uint8),(size,size), interpolation=cv2.INTER_NEAREST)
        #Ensures output is strictly 0/1
    return (mask01 > 0).astype(np.uint8)

#display ulcer mask overlay
def overlay_mask(rgb: np.ndarray,mask01:np.ndarray,alpha: float=0.35) -> np.ndarray:
    base = rgb.copy()
    over = base.copy()
    over[mask01 == 1]=np.array([255, 0, 0],dtype=np.uint8) # Colours pixels where mask01==1 as red
    return cv2.addWeighted(over, float(alpha), base, 1-float(alpha),0)

#draw a translucent grid overlay(this is used during calibration)
def draw_grid(rgb: np.ndarray, spacing_px: int,thickness: int = 1,opacity:float = 0.2) -> np.ndarray:
    h,w = rgb.shape[:2]
    grid = rgb.copy()
    color = (255, 255, 255)
    #this Draws vertical and horizontal white lines every spacing_px
    spacing_px=max(1, int(spacing_px))
    for x in range(0, w, spacing_px):
        cv2.line(grid, (x, 0), (x, h), color, int(thickness))
    for y in range(0, h, spacing_px):
        cv2.line(grid, (0, y), (w, y), color, int(thickness))
    # Blends grid with the original image using addWeighted
    return cv2.addWeighted(grid, float(opacity), rgb, 1 - float(opacity), 0)

#used to detect new image uploaded and update UI
def fileFingerprint(name: str, b: bytes) -> str:
    h=hashlib.sha256()
    h.update(name.encode("utf-8",errors="ignore"))
    h.update(b)
    return h.hexdigest()[:16]


def bytesToRgb(b: bytes) -> np.ndarray:
    img = Image.open(BytesIO(b)).convert("RGB")
    return np.array(img)


def uploaderToRgb(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

#calls overaly mask / it is used in pdf report generation 
def explainability_overlay(rgb_u8: np.ndarray,mask01:np.ndarray,alpha_mask: float = 0.35)->np.ndarray:
    return overlay_mask(rgb_u8,mask01,alpha=alpha_mask)


# IDs / input helpers
#used before editing DB
def SafeCaeID(case_id: str) -> str:
    case_id = (case_id or "").strip()
    return case_id if case_id else ""


# Dataset saving
#saves an anonymized training style sample
def SaveAnonymisedSample(rgb: np.ndarray, mask01: np.ndarray, meta: dict) -> str:
    sample_id= str(uuid.uuid4())
    sample_dir= os.path.join(DATASET_DIR, sample_id)
    os.makedirs(sample_dir,exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(sample_dir,"image.png"))
    Image.fromarray((mask01.astype(np.uint8)*255)).save(os.path.join(sample_dir,"mask.png"))
    with open(os.path.join(sample_dir,"meta.json"),"w",encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return sample_id


# PDF report
#Generate a one page PDF report (ReportLab)
def make_pdf_report(case_id:str,visit_date:str,eye:str,summary:dict,rgb:np.ndarray,mask:np.ndarray,report_text:str)->bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab not installed")
#create overaly image 
    fig = explainability_overlay(rgb, mask, alpha_mask=0.35)
    #convert overlay to PNG in memory (BytesIO) for embedding
    pil = Image.fromarray(fig)
    bio = BytesIO()
    pil.save(bio, format="PNG")
    bio.seek(0)

#create report lab camvas A4 
    buf = BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    w, h = A4
#PRINT OUT THE metrics 
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 50, "Corneal ulcer assessment (assistive)")

    c.setFont("Helvetica", 11)
    c.drawString(40, h - 75, f"Case: {case_id}")
    c.drawString(40, h - 92, f"Date: {visit_date}")
    c.drawString(40, h - 109, f"Eye: {eye}")
    c.drawString(40, h - 126, f"Modality: {summary.get('mode','')}")

    y=h - 160
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
    c.drawString(40,y-10,"Overlay")
    img=ImageReader(bio)

    #smaller image so there is space for the report text
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
        c.drawString(40, h - 126, f"Modality: {summary.get('mode','')}")
        y = h - 160
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40,y-10,"Overlay")
        img_y = y - 24 - img_h

    c.drawImage(img, 40, img_y, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")

    c.setFont("Helvetica-Bold",12)
    c.drawString(40,img_y-18,"Report")
    c.setFont("Helvetica",9)

    from reportlab.pdfbase.pdfmetrics import stringWidth

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

    text_obj = c.beginText(40, img_y-34)
    line_h = 11
    bottom = 50
    max_w = w - 80

    for raw in report_text.splitlines():
        for line in wrapLine(raw, max_w):
            if text_obj.getY() < bottom:
                c.drawText(text_obj)
                c.showPage()
                c.setFont("Helvetica",9)
                text_obj = c.beginText(40, h-175)
            text_obj.textLine(line)
    c.drawText(text_obj)

    c.save()
    buf.seek(0)
    return buf.read()


# Upload persistence
#This is to handle Streamlit reruns and keep the last upload
def remember_upload(ns:str,uploaded_file):
    if uploaded_file is None:
        return
    st.session_state[f"{ns}_last_name"]=uploaded_file.name
    st.session_state[f"{ns}_last_bytes"]=uploaded_file.getvalue()

#Reads those values back from session_state
def get_remembered_upload(ns: str) -> Tuple[Optional[str], Optional[bytes]]:
    return st.session_state.get(f"{ns}_last_name"), st.session_state.get(f"{ns}_last_bytes")


# fluorescent model
@st.cache_resource  # The model loads once per app session not every rerun
def load_Fluerescent_model(model_path:str):
    if not KERAS_OK:
        raise RuntimeError("Install TensorFlow/Keras to use Fluerescent model.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found:{model_path}")
    if hasattr(keras,"saving") and hasattr(keras.saving, "load_model"):
        return keras.saving.load_model(model_path, compile=False)  # type: ignore
    return keras.models.load_model(model_path, compile=False)  # type: ignore


def preprocess_for_keras(rgb_u8: np.ndarray)->np.ndarray:
    x = rgb_u8.astype(np.float32)/255.0 #Convert uint8 0...255 -> float32 0..1
    return np.expand_dims(x,axis=0) #Add batch dimension:shape becomes(1,H,W,3)

#Handles different model output formats:
#p= squeeze(pred) removes batch dims
#If p.ndim==3 take channel 0 (p[...,0])
# If values look like logits (outside ~[0,1])-> apply sigmoid
#Threshold -> binary mask
#Resizes mask to MODEL_SIZE (256) and ensure 0/1.
def postprocess_keras_output(pred: np.ndarray,threshold:float)->np.ndarray:
    p = np.squeeze(np.array(pred))
    if p.ndim == 3:
        p = p[..., 0]
    if p.min() < -0.01 or p.max() > 1.01:
        p = 1.0 / (1.0 + np.exp(-p))
    mask01 = (p >= float(threshold)).astype(np.uint8)
    return ensure_mask01_size(mask01, MODEL_SIZE)

#Runs preprocess -> model.predict()-> postprocess
def predict_Fluerescent_mask01(model, rgb_u8_256: np.ndarray, threshold: float) -> np.ndarray:
    x= preprocess_for_keras(rgb_u8_256)
    pred= model.predict(x, verbose=0)
    return postprocess_keras_output(pred,threshold)  #Returns a 256×256 binary mask


# White-light model
@st.cache_resource
def load_ulcer_unet_cached(ckpt_path: str):
    model,device=load_ulcer_unet(ckpt_path)
    return model,device


# Stepper state/ workflow state machine
def step_state(ns: str) -> int:
    return int(st.session_state.get(f"{ns}_step", 1))

def set_step(ns: str, step: int):
    st.session_state[f"{ns}_step"] = int(step)

def stepper_ui(ns: str):
    s = step_state(ns)

    def dot(on):  #noqa: E704
        return "on" if on else "off"

    st.markdown(
        f"""
        <div class="stepper">
          <div class="chip"><span class="dot {dot(s>=1)}"></span><b>1</b> Upload</div>
          <div class="chip"><span class="dot {dot(s>=2)}"></span><b>2</b> Adjust</div>
          <div class="chip"><span class="dot {dot(s>=3)}"></span><b>3</b> Confirm</div>
          <div class="chip"><span class="dot {dot(s>=4)}"></span><b>4</b> Calibrate</div>
          <div class="chip"><span class="dot {dot(s>=5)}"></span><b>5</b> Results</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Mask editor helpers
#These manage Streamlit’s state so editing works predictably across reruns
def mask_key(ns:str) -> str:  #where the current editable mask is stored
    return f"{ns}_mask01_display"


def canvas_nonce_key(ns:str) -> str: #a counter to force Streamlit to treat the canvas as new
    return f"{ns}_canvas_nonce"


def canvas_key(ns:str) -> str: # returns a unique canvas widget key based on the nonce
    nonce= int(st.session_state.get(canvas_nonce_key(ns), 0))
    return f"{ns}_mask_editor_canvas_{nonce}"

#increments nonce so the drawable canvas clears its strokes on next rerun
def reset_canvas(ns:str):
    st.session_state[canvas_nonce_key(ns)] = int(st.session_state.get(canvas_nonce_key(ns), 0)) + 1


#if user uploads a different image then clear all state tied to the previous one
def reset_editor_for_new_image(ns:str,fp:str):
    last_fp_key = f"{ns}_last_fp"
    if st.session_state.get(last_fp_key)!=fp:
        st.session_state[last_fp_key] = fp
        for k in [
            mask_key(ns),
            canvas_nonce_key(ns),
            f"{ns}_mask_confirmed",
            f"{ns}_mm_per_px",
            f"{ns}_ref_roi_xywh",
            f"{ns}_ref_rect_canvas",
            f"{ns}_linecanvas",
        ]:
            st.session_state.pop(k, None)
        set_step(ns, 2)

#If no editable mask exists yet initialize it from the prediction
def init_mask_state(ns: str, pred_mask01_display: np.ndarray):
    k=mask_key(ns)
    if k not in st.session_state:
        st.session_state[k] = pred_mask01_display.copy()

#Returns a PIL image that is the overlay background for the canvas
def make_editor_background(rgb_display:np.ndarray, mask01_display:np.ndarray, alpha:float) -> Image.Image:
    return Image.fromarray(overlay_mask(rgb_display, mask01_display, alpha=alpha))

#This is the 'apply brush changes' logic
def apply_strokes(canvas_rgba:np.ndarray, mask01_display:np.ndarray, mode:str) -> np.ndarray:
    if canvas_rgba is None:
        return mask01_display

    rgba=canvas_rgba.astype(np.uint8)

    #alpha channel => any painted pixel has alpha > 0
    if rgba.shape[-1]==4:
        a= rgba[..., 3]
        painted= a > 0
        new_mask= mask01_display.copy()
        if mode == "Add":
            new_mask[painted] = 1
        else:
            new_mask[painted] = 0
        return ensure_mask01_size(new_mask, DISPLAY_SIZE)

    #Fallback if no alpha channel
    r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    if mode == "Add":
        painted = (g > 150) & (r < 170) & (b < 170)
    else:
        painted = (r > 160) & (b > 160) & (g < 180)

    new_mask= mask01_display.copy()
    if mode == "Add":
        new_mask[painted] = 1
    else:
        new_mask[painted] = 0
    return ensure_mask01_size(new_mask, DISPLAY_SIZE)

#Ensure editable mask state exists.
# Show controls:
#       radio Add/Erase
#       brush size slider
#       overlay alpha slider
#       If drawable canvas missing:
#       show static overlay + info message
#       return current mask
#           Else:
#       background= overlay image (single visual)
#       stroke color green for Add and magenta for Erase
#       Buttons:
#       Apply changes:merges strokes into mask, resets canvas (clears strokes), st.rerun()
#       Reset to prediction: restores original prediction, resets canvas, st.rerun()

def ReviewEdit(ns:str, rgb_display:np.ndarray, pred_mask01_display:np.ndarray) -> np.ndarray:
    init_mask_state(ns, pred_mask01_display)
    edited_mask = st.session_state[mask_key(ns)]

    st.caption("If anything is missed or over-segmented, adjust the mask.")

    c1,c2,c3 = st.columns([1.2, 1.1, 1.2])
    with c1:
        mode= st.radio("Brush", ["Add", "Erase"], horizontal=True, key=f"{ns}_brush_mode")
    with c2:
        brush= st.slider("Size", 3, 70, int(st.session_state.get(f"{ns}_brushsize", 18)), 1, key=f"{ns}_brushsize")
    with c3:
        alpha = st.slider("Overlay", 0.05, 0.85, float(st.session_state.get(f"{ns}_alpha", 0.35)), 0.05, key=f"{ns}_alpha")

    if not CANVAS_OK:
        st.image(overlay_mask(rgb_display, edited_mask, alpha=alpha), **img_kw())
        st.info("Install streamlit-drawable-canvas to enable brush editing.")
        return edited_mask

    bg= make_editor_background(rgb_display, edited_mask, alpha=alpha)
    stroke_color = "#00E676" if mode == "Add" else "#FF4DFF"

    canvas_result= st_canvas(
        fill_color="rgba(255,255,255,0.0)",
        stroke_width=int(brush),
        stroke_color=stroke_color,
        background_image=bg,
        update_streamlit=True,
        height=DISPLAY_SIZE,
        width=DISPLAY_SIZE,
        drawing_mode="freedraw",
        key=canvas_key(ns),
    )

    cA,cB = st.columns(2)
    with cA:
        if st.button("Apply changes", key=f"{ns}apply_strokes"):
            if canvas_result.image_data is not None:
                new_mask = apply_strokes(canvas_result.image_data, edited_mask, mode=mode)
                st.session_state[mask_key(ns)] = new_mask
                reset_canvas(ns)  # critical: clear strokes so next apply only uses new drawing
                st.rerun()

    with cB:
        if st.button("Reset to prediction", key=f"{ns}_reset_mask"):
            st.session_state[mask_key(ns)] = pred_mask01_display.copy()
            reset_canvas(ns)
            st.rerun()

    return st.session_state[mask_key(ns)]



# Grey reference selector (white-light)
#normalize brightness for opacity metric comparability
def white_reference_selector(rgb_display:np.ndarray, ns:str):
    use_ref=st.checkbox(
        "Use a grey reference (if visible)",
        value=bool(st.session_state.get(f"{ns}_use_ref",False)),
        key=f"{ns}_use_ref",
    )
    target_grey= st.slider("Target grey level",80.0,170.0,float(st.session_state.get(f"{ns}_target_grey", 120.0)),1.0,key=f"{ns}_target_grey",)
    roi=st.session_state.get(f"{ns}_ref_roi_xywh", None)
    if not use_ref:
        return False, None, float(target_grey)

    st.markdown(
        """
**Why this helps:** white light slit lamp images can vary in brightness/exposure.  
If you mark a neutral grey patch, the app can normalise brightness so the **opacity proxy** is more comparable across visits.
"""
    )

    if not CANVAS_OK:
        st.warning("Install streamlit-drawable-canvas to select the grey patch.")
        return True, roi, float(target_grey)
    
    st.caption("Draw a rectangle around the grey patch.")
    bg = Image.fromarray(rgb_display)
    canvas = st_canvas(
        fill_color="rgba(255,255,255,0.0)",
        stroke_width=2,
        stroke_color="#00E676",
        background_image=bg,
        update_streamlit=True,
        height=DISPLAY_SIZE,
        width=DISPLAY_SIZE,
        drawing_mode="rect",
        key=f"{ns}_ref_rect_canvas",
    )

    if canvas.json_data and "objects" in canvas.json_data and len(canvas.json_data["objects"]) > 0:
        obj = canvas.json_data["objects"][-1]
        if obj.get("type")=="rect":
            x= int(obj.get("left", 0))
            y= int(obj.get("top", 0))
            w= int(obj.get("width", 0) * obj.get("scaleX", 1))
            h= int(obj.get("height", 0) * obj.get("scaleY", 1))
            if w>0 and h > 0:
                roi= (x, y, w, h)
                st.session_state[f"{ns}_ref_roi_xywh"]= roi

    if roi is not None:
        st.caption(f"Grey patch ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        if st.button("Clear grey patch selection", key=f"{ns}_clear_ref_roi"):
            st.session_state.pop(f"{ns}_ref_roi_xywh", None)
            roi = None
    else:
        st.info("No grey patch selected yet.")

    return True, roi, float(target_grey)


# Calibration
def calibration(ns:str, rgb_display:np.ndarray, mask01_display:np.ndarray):
    st.caption("Calibration sets the physical scale (mm/px). Use a real in frame reference if you can.")
    base = overlay_mask(rgb_display, mask01_display, alpha=0.35)
    method = st.radio("Method", ["Line", "Grid (research only)"], horizontal=True, key=f"{ns}_cal_method")
#choose a method line or grid 
    if method == "Line":
        st.caption("Draw a line across a known distance (eg: a ruler/marker in the image).")
        if not CANVAS_OK:
            st.image(base, **img_kw())  # fallback only
            st.error("Install streamlit-drawable-canvas to use line calibration.")
            return

        known_mm=st.number_input(
            "Known distance (mm)",
            min_value=0.01,
            value=float(st.session_state.get(f"{ns}_known_mm", 1.0)),
            step=0.1,
            key=f"{ns}_known_mm",
        )
        bg = Image.fromarray(draw_grid(base, 25, thickness=1, opacity=0.16))
#display the image
        line_canvas = st_canvas(
            fill_color="rgba(255,255,255,0.0)",
            stroke_width=3,
            stroke_color="#00E676",
            background_image=bg,
            update_streamlit=True,
            height=DISPLAY_SIZE,
            width=DISPLAY_SIZE,
            drawing_mode="line",
            key=f"{ns}_linecanvas",
        )

        if line_canvas.json_data and "objects" in line_canvas.json_data and len(line_canvas.json_data["objects"])> 0:
            obj= line_canvas.json_data["objects"][-1]
            if obj.get("type")== "line":
                x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
                px_dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
                if px_dist > 0:
                    st.session_state[f"{ns}_mm_per_px"] = float(known_mm) / px_dist
#grid logic 
    else:
        st.warning("Use grid only if the mm value is genuinely known (not guessed).")
        grid_mm= st.selectbox("Grid square size (mm)", [0.05, 0.1, 0.2, 0.5, 1.0], index=1, key=f"{ns}_gridmm")
        spacing_px= st.slider("Grid spacing (px)", 5, 200, 25, 1, key=f"{ns}_gridpx")
        st.session_state[f"{ns}_mm_per_px"]=float(grid_mm)/float(spacing_px)
        st.image(draw_grid(base, spacing_px, thickness=1, opacity=0.22), **img_kw())
    mm=st.session_state.get(f"{ns}_mm_per_px")
    st.metric("Calibration (mm/px)", "—" if mm is None else f"{mm:.6f}")

    if st.button("Confirm calibration", key=f"{ns}_confirm_cal"):
        if not (isinstance(mm, (int, float)) and float(mm) > 0):
            st.warning("Set a valid calibration first.")
        else:
            set_step(ns,5)


# Metrics + report + timeline (DB)
#Main results function 
def metrics(ns: str, rgb_display: np.ndarray, mask01_display: np.ndarray, compute_opacity: bool):
    mm_per_px=st.session_state.get(f"{ns}_mm_per_px", None)
    if not (isinstance(mm_per_px,(int,float)) and float(mm_per_px)>0):
        st.warning("Calibration not set (Go back to Step 4)")
        return

    eye=st.session_state.get("session_eye", "Right")
    case_id= SafeCaeID(st.session_state.get("case_id", ""))
    visit_date= (st.session_state.get("visit_date") or "").strip() or datetime.now().strftime("%Y-%m-%d")
    ref_roi= None
    ref_target= 120.0
    if ns=="white":
        with st.expander("Grey reference (white light only)", expanded=False):
            use_ref,roi,target= white_reference_selector(rgb_display, ns=ns)
            if use_ref and roi is not None:
                ref_roi=roi
                ref_target=float(target)

    result=analyse_arrays(
        rgb=rgb_display,
        mask01=mask01_display,
        case_id=case_id if case_id else None,
        visit_date=visit_date,
        mm_per_pixel=float(mm_per_px),
        source="verified",
        compute_opacity=compute_opacity,
        eye=eye,
        reference_roi_xywh=ref_roi,
        reference_target_grey=float(ref_target),
    )

    flags = result.get("analysis_flags") or []
#Display metrics in streamlit st.metric blocks 
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Area (mm²)", "N/A" if result.get("area_mm2") is None else f"{result['area_mm2']:.4f}")
    c2.metric("Eq diameter (mm)", "N/A" if result.get("eq_diameter_mm") is None else f"{result['eq_diameter_mm']:.2f}")
    c3.metric("Zone", str(result.get("zone", "—")))
    c4.metric("Blur", f"{result.get('blur_laplacian_var', 0.0):.2f}")

    loc1,loc2,loc3=st.columns(3)
    loc1.metric("Vertical", str(result.get("vertical_sector", "—")))
    loc2.metric("Horizontal", str(result.get("horizontal_sector", "—")))
    loc3.metric("Eye", eye)

    if compute_opacity:
        o1,o2,o3 = st.columns(3)
        o1.metric("Opacity mean", "—" if result.get("opacity_mean") is None else f"{result['opacity_mean']:.3f}")
        o2.metric("Opacity contrast", "—" if result.get("opacity_contrast") is None else f"{result['opacity_contrast']:.3f}")
        o3.metric("Opacity z-score", "—" if result.get("opacity_zscore") is None else f"{result['opacity_zscore']:.2f}")

    if flags:
        st.info("Quality notes: " + ", ".join(flags))

    #Builds a summary dict used for report generation
    b= st.session_state.get(f"{ns}_last_bytes")
    nm= st.session_state.get(f"{ns}_last_name", "image")
    img_hash= fileFingerprint(nm, b) if isinstance(b, (bytes, bytearray)) else "unknown"

    summary={
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

    st.divider()
    st.markdown("### Report")

    acq_notes=st.text_area(
        "Notes (optional)",
        value=st.session_state.get(f"{ns}_acq_notes", ""),
        key=f"{ns}_acq_notes",
        height=80,
    )

    llm_result= generate_report_with_llm(summary,acquisition_notes=acq_notes)
    report_text=llm_result.report_text

    #st.caption(f"Report source: {'LLM' if llm_result.used_llm else 'Template'} • Model: {llm_result.used_model}")

    if acq_notes.strip():
        report_text = report_text.rstrip() + "\n\nClinician notes\n" + acq_notes.strip()
    st.text(report_text)

    # Download
    #If ReportLab OK-> try PDF 
    #If PDF fails or reportlab missing -> download .txt
    if REPORTLAB_OK:
        try:
            pdf_bytes= make_pdf_report(case_id or "CASE", visit_date, eye, summary, rgb_display, mask01_display, report_text)
            st.download_button(
                "Download report",
                data=pdf_bytes,
                file_name=f"{(case_id or 'CASE')}_{visit_date}_{eye}_report.pdf",
                mime="application/pdf",
            )
        except Exception:
            st.download_button(
                "Download report",
                data=report_text.encode("utf-8"),
                file_name=f"{(case_id or 'CASE')}_{visit_date}_{eye}_report.txt",
                mime="text/plain",
            )
    else:
        st.download_button(
            "Download report",
            data=report_text.encode("utf-8"),
            file_name=f"{(case_id or 'CASE')}_{visit_date}_{eye}_report.txt",
            mime="text/plain",
        )

    st.divider()
    st.markdown("### Timeline")

#if no case id disable saving 
    if not case_id:
        st.info("Add a Case ID to save this visit to the timeline.")
        save_enabled = False
    else:
        save_enabled = True

    visit_id = hashlib.sha1(f"{case_id}|{visit_date}|{eye}|{ns}|{img_hash}".encode("utf-8")).hexdigest()[:16]

    if st.button("Save this visit", disabled=not save_enabled):
        try:
            save_visit(
                visit_id=visit_id,
                case_id=case_id,
                visit_date=visit_date,
                eye=eye,
                mode=ns,
                image_hash=img_hash,
                mm_per_px=float(mm_per_px),
                analysis_result=result,
                report_text=report_text,
            )
            st.success("Saved to database.")
        except Exception as e:
            st.error(f"Save failed: {e}")
#load all case visits realted to this one 
    visits = load_case_visits(case_id) if case_id else []
    if not visits:
        return

    case_df = pd.DataFrame(
        [
            {
                "visit_date": v.visit_date,
                "eye": v.eye,
                "mode": v.mode,
                "area_mm2": v.area_mm2,
                "eq_diameter_mm": v.eq_diameter_mm,
                "opacity_zscore": v.opacity_zscore,
                "mm_per_px": v.mm_per_px,
                "qc_flags": v.qc_flags,
                "created_at": v.created_at,
            }
            for v in visits
        ]
    )

    case_df["visit_date"]=pd.to_datetime(case_df["visit_date"], errors="coerce")
    case_df=case_df.sort_values("visit_date")
    st.dataframe(case_df, use_container_width=True)


    # Trends (best practice here: 2 charts when opacity exists otherwise 1)

    valid_dates = case_df["visit_date"].dropna()
    #if more than 2 visits plot the graphs 
    if len(valid_dates) >= 2:
        st.markdown("#### Trends")

        # Clean+sort for correct X-axis direction
        case_df= case_df.dropna(subset=["visit_date"]).sort_values("visit_date")

        for col in ["area_mm2", "eq_diameter_mm", "opacity_zscore", "mm_per_px"]:
            if col in case_df.columns:
                case_df[col]=pd.to_numeric(case_df[col],errors="coerce")

        #Explanation+warnings
        exp_col,charts_col=st.columns([1, 2])

        with exp_col:
            st.markdown(
                """
                        **How to read the charts**
                        - **X-axis:** visit date  
                        - **Y-axis:** metric value  
                        - Lines connect visits in chronological order.
                        - **Area/diameter** reflect size change.
                        - **Opacity z-score** is a **proxy** (white light only)/sensitive to lighting and focus.
                """
            )

            warnings= []
            mm= case_df.get("mm_per_px", pd.Series(dtype=float)).dropna()
            if len(mm) >= 2:
                mm_min,mm_max= float(mm.min()), float(mm.max())
                if mm_min > 0:
                    pct = (mm_max - mm_min) / mm_min * 100.0
                    if pct >= 10.0:
                        warnings.append(f"Calibration (mm/px) varies by ~{pct:.1f}% across visits.")

            qc= case_df.get("qc_flags",pd.Series(dtype=str)).fillna("").astype(str).str.strip()
            if (qc!="").any():
                warnings.append("Some visits have QC flags: changes may reflect image quality not anatomy.")
            if warnings:
                st.warning("Comparability notes:\n- " + "\n- ".join(warnings))
            else:
                st.success("Visits look broadly comparable (based on calibration + QC).")

            # Quick delta summary (last vs previous)
            # st.markdown("**Latest vs previous**")
            # def _delta_block(label: str, col: str):
            #     s = case_df[["visit_date", col]].dropna()
            #     if len(s) >= 2:
            #         prev_val = float(s.iloc[-2][col])
            #         last_val = float(s.iloc[-1][col])
            #         delta = last_val - prev_val
            #         pct = (delta / prev_val * 100.0) if prev_val != 0 else np.nan
            #         st.write(f"- {label}: **{last_val:.4g}**  (Δ **{delta:+.4g}**" + (f", **{pct:+.1f}%**)" if np.isfinite(pct) else ")"))
            #     elif len(s) == 1:
            #         st.write(f"- {label}: **{float(s.iloc[-1][col]):.4g}**")
            #     else:
            #         st.write(f"- {label}: —")

            # if "area_mm2" in case_df.columns:
            #     _delta_block("Area (mm²)", "area_mm2")
            # if "eq_diameter_mm" in case_df.columns:
            #     _delta_block("Eq diameter (mm)", "eq_diameter_mm")
            # if "opacity_zscore" in case_df.columns and case_df["opacity_zscore"].notna().any():
            #     _delta_block("Opacity z-score", "opacity_zscore")

        with charts_col:
            plot_df=case_df.set_index("visit_date").sort_index()

            # Chart 1: geometry (area + diameter)
            geom_cols = [c for c in ["area_mm2", "eq_diameter_mm"] if c in plot_df.columns and plot_df[c].notna().any()]
            if geom_cols:
                st.markdown("**Size metrics**")
                st.line_chart(plot_df[geom_cols])

            # Chart 2: opacity proxy (only if present and meaningful)
            if "opacity_zscore" in plot_df.columns and plot_df["opacity_zscore"].notna().sum() >= 2:
                st.markdown("**Opacity proxy (white-light)**")
                st.line_chart(plot_df[["opacity_zscore"]])

# Prediction pipelines
def run_Fluerescent_prediction()->Tuple[np.ndarray, np.ndarray]:
    ns= "Fluerescent"
    up= st.file_uploader("Fluerescent image", type=["png", "jpg", "jpeg"], key="Fluerescent_uploader")
    remember_upload(ns, up)

    model_path= os.environ.get("Fluerescent_MODEL_PATH", os.path.join(APP_DIR, "RealDataModelv2.keras"))
    thr= st.slider("Threshold", 0.05, 0.95, float(st.session_state.get("Fluerescent_thr", 0.50)), 0.05, key="Fluerescent_thr")
    if up is None:
        name,b = get_remembered_upload(ns)
        if b is None:
            raise RuntimeError("No Fluerescent image uploaded yet.")
        rgb_native = bytesToRgb(b)
        img_name, img_bytes = name, b
    else:
        img_bytes= up.getvalue()
        img_name= up.name
        rgb_native= uploaderToRgb(up)

    fp= fileFingerprint(img_name, img_bytes)
    reset_editor_for_new_image(ns, fp)

    rgb_display= ensureRgbSize(rgb_native, DISPLAY_SIZE)
    rgb_model= ensureRgbSize(rgb_native, MODEL_SIZE)

    pred_key= f"Fluerescent|{fp}|thr={thr:.3f}|model={model_path}"

    if st.session_state.get("Fluerescent_pred_key")!=pred_key:
        model= load_Fluerescent_model(model_path)
        pred_mask01_model= predict_Fluerescent_mask01(model, rgb_model, threshold=thr)
        st.session_state["Fluerescent_pred_mask01_model"]= pred_mask01_model
        st.session_state["Fluerescent_pred_key"]= pred_key

    pred_mask01_display= ensure_mask01_size(st.session_state["Fluerescent_pred_mask01_model"], DISPLAY_SIZE)

    if step_state(ns) < 2:
        set_step(ns, 2)

    return rgb_display,pred_mask01_display


def run_white_prediction()->Tuple[np.ndarray, np.ndarray]:
    ns= "white"
    up= st.file_uploader("White light image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="white_uploader")
    remember_upload(ns, up)

    ckpt = os.environ.get("WHITE_CKPT_PATH", os.path.join(APP_DIR, "best.pt"))
    thr= st.slider("Threshold", 0.05, 0.95, float(st.session_state.get("white_thr", 0.50)), 0.05, key="white_thr")
    if up is None:
        name, b = get_remembered_upload(ns)
        if b is None:
            raise RuntimeError("No white light image uploaded yet.")
        img_name, img_bytes = name, b
        suffix= os.path.splitext(name)[1].lower()
        if suffix not in [".png",".jpg", ".jpeg",".tif",".tiff"]:
            suffix = ".png"
    else:
        img_name= up.name
        img_bytes= up.getvalue()
        suffix= os.path.splitext(up.name)[1].lower()
        if suffix not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            suffix = ".png"

    fp= fileFingerprint(img_name, img_bytes)
    reset_editor_for_new_image(ns, fp)

    pred_key= f"white|{fp}|thr={thr:.3f}|ckpt={ckpt}"

    if st.session_state.get("white_pred_key") != pred_key:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(img_bytes)
            tmp_path= tmp.name
        try:
            torch_model,torch_device= load_ulcer_unet_cached(ckpt)
            rgb_u8_512, mask01_512 = predict_mask_from_path(torch_model, torch_device, tmp_path, thr=thr)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        st.session_state["white_pred_rgb_512"] = rgb_u8_512
        st.session_state["white_pred_mask01_512"] = mask01_512
        st.session_state["white_pred_key"] = pred_key

    rgb_display= ensureRgbSize(st.session_state["white_pred_rgb_512"], DISPLAY_SIZE)
    pred_mask01_display= ensure_mask01_size(st.session_state["white_pred_mask01_512"], DISPLAY_SIZE)
    if step_state(ns) < 2:
        set_step(ns, 2)

    return rgb_display, pred_mask01_display

# App
st.set_page_config(page_title="Corneal Ulcer Assessment", layout="wide")
inject_css()
st.session_state.setdefault("case_id", "")
st.session_state.setdefault("visit_date", "")
st.session_state.setdefault("mode", None)
st.session_state.setdefault("session_eye", "Right")

st.markdown(
    """
    <div class="appbar">
      <div>
        <div class="brand">Corneal Ulcer Assessment</div>
        <div class="subtitle">Segmentation • Calibration • Measurements</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_workflow,tab_guide= st.tabs(["Workflow", "Guide"])

with tab_guide:
    st.markdown(
        """
### Practical notes
-1- Try to keep focus sharp and minimise glare.


-2- If you want mm² and mm to mean anything you need a real calibration reference (line over known distance is best).

### Grey reference (white light)
Use the grey patch when you're comparing **opacity proxy** across visits.


It helps reduce day to day lighting differences but it won't fix heavy glare or poor focus.
"""
    )

with tab_workflow:
    with st.sidebar:
        st.markdown("### Session")
        st.text_input("Case ID", value=st.session_state.get("case_id",""),key="case_id")
        st.text_input("Visit date (YYYY-MM-DD)",value=st.session_state.get("visit_date", ""),key="visit_date")
        st.selectbox(
            "Eye",
            ["Right","Left"],
            index=0 if st.session_state.get("session_eye", "Right") == "Right" else 1,
            key="session_eye",
        )
        st.divider()
        mode_choice= st.radio(
            "Image type",["Fluerescent", "White-light"],index=0 if st.session_state.get("mode", "Fluerescent") != "white" else 1,)
        if st.button("Open"):
            st.session_state["mode"] = "white" if mode_choice == "White-light" else "Fluerescent"
            st.rerun()

    if st.session_state["mode"] is None:
        st.info("Choose an image type in the sidebar to begin.")
        st.stop()

    mode = st.session_state["mode"]
    ns = "white" if mode == "white" else "Fluerescent"
    is_white = (ns == "white")

    if f"{ns}_step" not in st.session_state:
        set_step(ns, 1)

    left, right = st.columns([2.2, 1])

    with right:
        st.markdown('<div class="sticky">', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f'<div class="panelTitle">{"White-light" if is_white else "Fluerescent"}</div>', unsafe_allow_html=True)
        stepper_ui(ns)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with left:
        st.markdown(f"## {'White-light' if is_white else 'Fluerescent'}")
        st.markdown("### /1/ Upload")
        try:
            if is_white:
                rgb_display, pred_mask01_display = run_white_prediction()
            else:
                rgb_display, pred_mask01_display = run_Fluerescent_prediction()
        except Exception as e:
            st.error(str(e))
            st.stop()

        st.divider()
        st.markdown("### /2/ Adjust mask")
        edited_mask = ReviewEdit(ns, rgb_display, pred_mask01_display)

        st.divider()
        st.markdown("### /3/ Confirm")
        st.image(overlay_mask(rgb_display, edited_mask, alpha=0.35), **img_kw())
        if st.button("Use this mask", key=f"{ns}_confirm_mask"):
            set_step(ns, 4)
            st.session_state[f"{ns}_mask_confirmed"] = edited_mask.copy()

        st.divider()
        st.markdown("### /4/ Calibrate")
        if step_state(ns) < 4:
            st.info("Confirm the mask to continue.")
            st.stop()

        mask_for_cal = st.session_state.get(f"{ns}_mask_confirmed", edited_mask)
        calibration(ns, rgb_display, mask_for_cal)

        st.divider()
        st.markdown("### /5/ Results")
        if step_state(ns) < 5:
            st.info("Confirm calibration to continue.")
            st.stop()

        mask_for_metrics= st.session_state.get(f"{ns}_mask_confirmed", edited_mask)
        metrics(ns,rgb_display, mask_for_metrics, compute_opacity=is_white)