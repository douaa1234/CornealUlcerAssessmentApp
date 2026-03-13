from __future__ import annotations

import numpy as np
import cv2

# Basic helpers
#blur/sharpness prox
#highlights high frequency detail (edges).
def variance_of_laplacian(grey_u8: np.ndarray) -> float:
    return float(cv2.Laplacian(grey_u8, cv2.CV_64F).var())

#normalize local contrast (helps edges/circle detection and opacity proxy)
def clahe_grey(grey_u8: np.ndarray) -> np.ndarray:
    clahe= cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
    return clahe.apply(grey_u8)


def to_u8(x:np.ndarray) -> np.ndarray:
    if x.dtype==np.uint8:
        return x
    x=np.clip(x, 0, 255)
    return x.astype(np.uint8)


def ensure_mask01(mask01:np.ndarray) -> np.ndarray:
    return (mask01 > 0).astype(np.uint8)

# Grey reference correction (WHITE LIGHT standardisation)
#white light 'standardization' using a neutral grey patch ROI
def apply_grey_reference_correction(
    rgb_u8: np.ndarray,
    ref_roi_xywh: tuple[int, int, int, int],
    *,
    target_grey: float = 120.0,
):
    """
    Standardisation using a neutral grey patch:
      - mean RGB in ROI
      - per-channel gain so ROI ~ target_grey

    Returns: corrected_rgb_u8, debug_dict, qc_flags(list)
    """
    flags: list[str] = []

    rgb_u8 = to_u8(rgb_u8)
    H, W = rgb_u8.shape[:2]
    x, y, w, h = [int(v) for v in ref_roi_xywh]
#Clip ROI to image bounds, clamp w,h to at least 1 pixel
    x = int(np.clip(x, 0, W - 1))
    y = int(np.clip(y, 0, H - 1))
    w = int(np.clip(w, 1, W - x))
    h = int(np.clip(h, 1, H - y))
#QC checks on ROI:
#(REF_ROI_TOO_SMALL) if ROI < 200 pixels
    if w*h < 200:
        flags.append("REF_ROI_TOO_SMALL")

    patch= rgb_u8[y:y + h, x:x + w].astype(np.float32)
    flat= patch.reshape(-1, 3)
    mean_rgb= flat.mean(axis=0)
    mR,mG,mB=[float(v) for v in mean_rgb]

#Detect saturation in patch
    sat_frac= float((flat.max(axis=1) >= 250).mean()) if flat.size else 0.0
    if sat_frac > 0.02:
        flags.append("REF_ROI_SATURATED")

    grey_patch= cv2.cvtColor(to_u8(patch.astype(np.uint8)), cv2.COLOR_RGB2GRAY)
    if float(grey_patch.std()) < 3.0:
        flags.append("REF_ROI_LOW_DYNAMIC_RANGE")

    def neutrality_err(r, g, b):
        return float(abs(r - g) + abs(g - b) + abs(r - b))

    neutral_before = neutrality_err(mR, mG, mB)
    if neutral_before > 25.0:
        flags.append("REF_ROI_NOT_NEUTRAL_BEFORE")

    eps = 1e-6
    sR = float(target_grey / (mR + eps))
    sG = float(target_grey / (mG + eps))
    sB = float(target_grey / (mB + eps))

    corrected = rgb_u8.astype(np.float32)
    corrected[..., 0]*= sR
    corrected[..., 1]*= sG
    corrected[..., 2]*= sB
    corrected= np.clip(corrected, 0, 255).astype(np.uint8)

    patch_after= corrected[y:y + h, x:x + w].astype(np.float32).reshape(-1, 3)
    mean_after= patch_after.mean(axis=0)
    aR,aG,aB= [float(v) for v in mean_after]
    neutral_after= neutrality_err(aR, aG, aB)
    if neutral_after > 15.0:
        flags.append("REF_ROI_NOT_NEUTRAL_AFTER")

    dbg={
        "ref_roi_xywh": [x, y, w, h],
        "target_grey": float(target_grey),
        "ref_mean_rgb_before": [mR, mG, mB],
        "ref_mean_rgb_after": [aR, aG, aB],
        "scale_rgb": [sR, sG, sB],
        "ref_sat_frac": sat_frac,
        "ref_neutrality_err_before": neutral_before,
        "ref_neutrality_err_after": neutral_after,
    }
    return corrected, dbg, flags

# Area + diameter
def compute_area(mask01:np.ndarray, mm_per_pixel:float | None):
    mask01= ensure_mask01(mask01)
    area_px= int(mask01.sum())
    area_mm2= None if mm_per_pixel is None else float(area_px * (float(mm_per_pixel) ** 2))
    return {"area_px": area_px, "area_mm2": area_mm2}

#computes 'equivalent circle diameter' from area
def compute_equivalent_diameter(area_mm2:float | None):
    if area_mm2 is None or area_mm2 <= 0:
        return {"eq_diameter_mm": None}
    d = float(2.0*np.sqrt(area_mm2 / np.pi))
    return {"eq_diameter_mm": d}

# Contours + geometry
#finding the largest connected lesion blob
def largest_contour(mask01: np.ndarray):
    m= (mask01.astype(np.uint8) * 255)
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

#major/minor axis lengths via ellipse fit
#I need at least 5 points else none 
#This is a proxy for lesion 'length/width' in pixels
def ellipse_axes_from_contour(cnt):
    if cnt is None or len(cnt) < 5:
        return None, None
    try:
        (cx, cy), (a, b), _angle = cv2.fitEllipse(cnt)
        major = float(max(a, b))
        minor = float(min(a, b))
        return major, minor
    except Exception:
        return None, None

#Corneal centre + radius (limbus estimate)
#This code I got from the internet 
def estimate_corneal_circle(rgb_u8: np.ndarray, *, debug: bool = False):
    dbg = {}
    grey= cv2.cvtColor(to_u8(rgb_u8), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grey, (7, 7), 0)
    eq = clahe_grey(blur)
    edges = cv2.Canny(eq, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    h, w = grey.shape[:2]
    dbg["edges_nonzero"] = int((edges > 0).sum())

    minR = int(min(h, w) * 0.25)
    maxR = int(min(h, w) * 0.48)

    circles = cv2.HoughCircles(
        eq,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * 0.4),
        param1=120,
        param2=30,
        minRadius=minR,
        maxRadius=maxR,
    )
    if circles is not None and len(circles) > 0:
        circles = circles[0].astype(np.float32)
        icx, icy = w / 2.0, h / 2.0
        d = np.hypot(circles[:, 0] - icx, circles[:, 1] - icy)
        best = circles[int(np.argmin(d))]
        cx, cy, r = float(best[0]), float(best[1]), float(best[2])
        dbg["hough_candidates"] = int(circles.shape[0])
        dbg["hough_best"]= {"cx": cx, "cy": cy, "r": r}
        return cx, cy, r, "hough", dbg

    cx, cy = w / 2.0, h / 2.0
    r = float(min(cx, cy, w - cx, h - cy))
    dbg["fallback"] = {"cx": cx, "cy": cy, "r": r}
    return cx, cy, r, "fallback_image_center", dbg

#map lesion distance to clinical zones by normalized radius
def clinical_zone_from_distance(dist_px:float, cornea_r_px:float):
    cornea_r_px= max(float(cornea_r_px), 1.0)
    dn= float(dist_px / cornea_r_px)
    if dn < 1.0 / 3.0:
        return "central", dn
    if dn < 2.0 / 3.0:
        return "paracentral", dn
    return "peripheral", dn


# Quadrants (superior/inferior + nasal/temporal)
def lesion_quadrant(cx:float, cy:float, ccx:float, ccy:float, eye:str="Right"):
    dx= float(cx - ccx)
    dy= float(cy - ccy)
    vertical = "superior" if dy < 0 else "inferior" if dy > 0 else "central"

    if eye.lower().startswith("r"):
        horizontal = "nasal" if dx < 0 else "temporal" if dx > 0 else "central"
    else:
        horizontal = "temporal" if dx < 0 else "nasal" if dx > 0 else "central"
    return {
        "dx_from_centre_px": dx,
        "dy_from_centre_px": dy,
        "vertical_sector": vertical,
        "horizontal_sector": horizontal,
    }

# Position metrics
#compute centroid, bbox, axes, distance from center, zone, quadrant and cornea estimate
def position_metrics(
    mask01:np.ndarray,
    rgb_u8:np.ndarray | None= None,*,
    mm_per_pixel:float | None= None,
    cornea_center_xy:tuple[float, float] | None = None,
    cornea_radius_px:float | None= None,
    eye: str= "Right",
):
    mask01=ensure_mask01(mask01)
    h,w=mask01.shape[:2]
    cnt=largest_contour(mask01)

    if cnt is None:
        return {
            "centroid_x_px": None, "centroid_y_px": None,
            "bbox_x_px": None, "bbox_y_px": None, "bbox_w_px": None, "bbox_h_px": None,
            "major_axis_px": None, "minor_axis_px": None,
            "major_axis_mm": None, "minor_axis_mm": None,
            "dist_from_centre_px": None, "dist_from_centre_mm": None,
            "dist_norm": None,
            "zone": None,
            "vertical_sector": None,
            "horizontal_sector": None,
            "dx_from_centre_px": None,
            "dy_from_centre_px": None,
            "cornea_centre_x_px": None, "cornea_centre_y_px": None,
            "cornea_radius_px": None,
            "cornea_estimation_method": None,
        }

    x,y,bw,bh=cv2.boundingRect(cnt)

    M=cv2.moments(cnt)
    if M["m00"] > 0:
        cx=float(M["m10"] / M["m00"])
        cy=float(M["m01"] / M["m00"])
    else:
        cx=float(x + bw / 2.0)
        cy=float(y + bh / 2.0)

    major_px, minor_px = ellipse_axes_from_contour(cnt)

    if cornea_center_xy is not None and cornea_radius_px is not None:
        ccx, ccy = float(cornea_center_xy[0]), float(cornea_center_xy[1])
        cr= float(cornea_radius_px)
        cornea_method= "provided"
    else:
        if rgb_u8 is not None:
            ccx, ccy, cr, cornea_method, _ = estimate_corneal_circle(rgb_u8, debug=False)
        else:
            ccx, ccy = w / 2.0, h / 2.0
            cr = float(min(ccx, ccy, (w - ccx), (h - ccy)))
            cornea_method = "fallback_image_center"

    dist_px= float(np.hypot(cx - ccx, cy - ccy))
    dist_mm= None if mm_per_pixel is None else float(dist_px * float(mm_per_pixel))
    zone,dist_norm = clinical_zone_from_distance(dist_px, cr)

    quad= lesion_quadrant(cx, cy, ccx, ccy, eye=eye)

    return {
        "centroid_x_px": cx, "centroid_y_px": cy,
        "bbox_x_px": int(x), "bbox_y_px": int(y), "bbox_w_px": int(bw), "bbox_h_px": int(bh),
        "major_axis_px": major_px, "minor_axis_px": minor_px,
        "major_axis_mm": None if (mm_per_pixel is None or major_px is None) else float(major_px * float(mm_per_pixel)),
        "minor_axis_mm": None if (mm_per_pixel is None or minor_px is None) else float(minor_px * float(mm_per_pixel)),
        "dist_from_centre_px": dist_px,
        "dist_from_centre_mm": dist_mm,
        "dist_norm": float(dist_norm),
        "zone": zone,
        "vertical_sector": quad["vertical_sector"],
        "horizontal_sector": quad["horizontal_sector"],
        "dx_from_centre_px": quad["dx_from_centre_px"],
        "dy_from_centre_px": quad["dy_from_centre_px"],
        "cornea_centre_x_px": float(ccx),
        "cornea_centre_y_px": float(ccy),
        "cornea_radius_px": float(cr),
        "cornea_estimation_method": cornea_method,
    }


#Opacity proxy (WHITE LIGHT ONLY)
#ComputeS an intensity based 'opacity' metric using grayscale intensity in lesion vs surrounding ring
#brighter lesion than surroundings -> positive contrast.here we assumes whiter/brighter lesion implies more opacity/scar/ulcer reflectance.
def compute_opacity_white_light(rgb_u8:np.ndarray, mask01:np.ndarray):
    mask01= ensure_mask01(mask01)
    grey= cv2.cvtColor(to_u8(rgb_u8), cv2.COLOR_RGB2GRAY)
    norm= clahe_grey(grey)
    ulcer_px=norm[mask01==1]
    if ulcer_px.size== 0:
        return {
            "opacity_mean": None,
            "opacity_ring_mean": None,
            "opacity_contrast": None,
            "opacity_zscore": None,
            "opacity_ring_std": None,
            "opacity_flags": ["EMPTY_MASK"],
        }
    flags:list[str]=[]

    hi= np.percentile(ulcer_px, 99)
    valid =  ulcer_px[ulcer_px < hi] if ulcer_px.size > 50 else ulcer_px
    if valid.size< 10:
        flags.append("LOW_VALID_PIXELS")
        valid=ulcer_px

    mean_ulcer= float(valid.mean())

    kernel= np.ones((15, 15), np.uint8)
    dil= cv2.dilate(mask01, kernel, iterations=1)
    ring=(dil == 1) & (mask01 == 0)
    ring_px= norm[ring]

    if ring_px.size<30:
        flags.append("NO_RING_CONTEXT")
        return {
            "opacity_mean": mean_ulcer,
            "opacity_ring_mean": None,
            "opacity_contrast": None,
            "opacity_zscore": None,
            "opacity_ring_std": None,
            "opacity_flags": flags,
        }

    mean_ring = float(ring_px.mean())
    std_ring = float(ring_px.std() + 1e-6)
    contrast = float(mean_ulcer - mean_ring)
    z = float(contrast / std_ring)

    ulcer_grey = grey[mask01 == 1]
    sat_frac = float((ulcer_grey >= 250).mean()) if ulcer_grey.size else 0.0
    if sat_frac > 0.02:
        flags.append("HIGH_SPECULAR_GLARE")

    return {
        "opacity_mean": mean_ulcer,
        "opacity_ring_mean": mean_ring,
        "opacity_contrast": contrast,
        "opacity_zscore": z,
        "opacity_ring_std": std_ring,
        "opacity_flags": flags,
    }

# Quality control flags
def qc_flags(rgb_u8:np.ndarray, mask01:np.ndarray, mm_per_pixel:float | None):
    mask01 = ensure_mask01(mask01)
    grey= cv2.cvtColor(to_u8(rgb_u8), cv2.COLOR_RGB2GRAY)
    blur= variance_of_laplacian(grey)
    flags:list[str] = []
    if blur < 60:
        flags.append("POSSIBLE_BLUR")
    if mm_per_pixel is None:
        flags.append("NO_CALIBRATION_MM_PER_PIXEL")
    area_px= int(mask01.sum())
    if area_px== 0:
        flags.append("MASK_EMPTY")
    elif area_px< 50:
        flags.append("TINY_MASK_CHECK_THRESHOLD")

    return blur, flags

# Main API
#THis runs the whole analysis and returns a single flat results dictionary
def analyse_arrays(
    rgb: np.ndarray,
    mask01: np.ndarray,
    *,
    case_id=None,
    visit_date=None,
    mm_per_pixel: float | None = None,
    source: str = "verified",
    compute_opacity: bool = False,
    cornea_center_xy: tuple[float, float] | None = None,
    cornea_radius_px: float | None = None,
    eye: str = "Right",
    reference_roi_xywh: tuple[int, int, int, int] | None = None,
    reference_target_grey: float = 120.0,
):
    rgb_u8= to_u8(rgb)
    mask01= ensure_mask01(mask01)

    ref_dbg= None
    ref_flags:list[str] = []
    if reference_roi_xywh is not None:
        rgb_u8,ref_dbg,ref_flags= apply_grey_reference_correction(
            rgb_u8,reference_roi_xywh,target_grey=float(reference_target_grey)
        )

    area= compute_area(mask01, mm_per_pixel=mm_per_pixel)
    eqd= compute_equivalent_diameter(area.get("area_mm2"))

    pos= position_metrics(
        mask01,
        rgb_u8=rgb_u8,
        mm_per_pixel=mm_per_pixel,
        cornea_center_xy=cornea_center_xy,
        cornea_radius_px=cornea_radius_px,
        eye=eye,
    )

    blur,flags= qc_flags(rgb_u8, mask01, mm_per_pixel=mm_per_pixel)

    if compute_opacity:
        op= compute_opacity_white_light(rgb_u8, mask01)
        flags= sorted(set(flags + (op.get("opacity_flags") or [])))
    else:
        op= {
            "opacity_mean": None,
            "opacity_ring_mean": None,
            "opacity_contrast": None,
            "opacity_zscore": None,
            "opacity_ring_std": None,
        }

    if ref_flags:
        flags  = sorted(set(flags + ref_flags))

    return {
        "case_id": case_id,
        "visit_date": visit_date,
        "source": source,
        "mm_per_pixel": mm_per_pixel,
        "eye": eye,
        **area,
        **eqd,
        **pos,
        **op,
        "reference_used": reference_roi_xywh is not None,
        "reference_target_grey": float(reference_target_grey) if reference_roi_xywh is not None else None,
        "reference_debug": ref_dbg,
        "blur_laplacian_var": float(blur),
        "analysis_flags": flags,
    }
