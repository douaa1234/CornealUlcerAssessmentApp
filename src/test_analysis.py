import numpy as np
import pytest
from analysis import (
    compute_area,
    compute_equivalent_diameter,
    variance_of_laplacian,
    ensure_mask01,
    clinical_zone_from_distance,
    lesion_quadrant,
)

#Area Tests

def test_area_pixel_count():
    # 10x10 mask of all ones = 100 pixels
    mask = np.ones((10, 10), dtype=np.uint8)
    result = compute_area(mask, mm_per_pixel=None)
    assert result["area_px"] == 100

def test_area_mm2_correct():
    # 100 pixels at 0.1 mm/px = 100 * 0.01 = 1.0 mm2
    mask = np.ones((10, 10), dtype=np.uint8)
    result = compute_area(mask, mm_per_pixel=0.1)
    assert abs(result["area_mm2"] - 1.0) < 0.0001

def test_area_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    result = compute_area(mask, mm_per_pixel=0.1)
    assert result["area_px"] == 0
    assert result["area_mm2"] == 0.0

def test_area_no_callibration_returns_none():
    mask = np.ones((10, 10), dtype=np.uint8)
    result = compute_area(mask, mm_per_pixel=None)
    assert result["area_mm2"] is None

# Equivalent diameter tests
def test_eq_diameter_none_when_no_area():
    result = compute_equivalent_diameter(None)
    assert result["eq_diameter_mm"] is None

def test_eq_diameter_none_when_zero():
    result = compute_equivalent_diameter(0.0)
    assert result["eq_diameter_mm"] is None

def test_eq_diameter_correct():
    #Circle of area pi should have diameter 2
    import math
    result = compute_equivalent_diameter(math.pi)
    assert abs(result["eq_diameter_mm"] - 2.0) < 0.0001

def test_eq_diameter_positive():
    result = compute_equivalent_diameter(4.0)
    assert result["eq_diameter_mm"] > 0

# ensure_mask01 tests

def test_ensure_mask01_binary():
    mask = np.array([[0, 128, 255]], dtype=np.uint8)
    result = ensure_mask01(mask)
    assert set(result.flatten().tolist()).issubset({0, 1})

def test_ensure_mask01_all_zeros():
    mask = np.zeros((5, 5), dtype=np.uint8)
    result = ensure_mask01(mask)
    assert result.sum() == 0

def test_ensure_mask01_all_ones():
    mask= np.ones((5, 5), dtype=np.uint8) * 255
    result= ensure_mask01(mask)
    assert result.sum()==25

# ClinicalZoneTests

def test_zone_central():
    zone, dn= clinical_zone_from_distance(10.0, 100.0)
    assert zone == "central"
    assert dn < 1/3

def test_zone_paracentral():
    zone, dn = clinical_zone_from_distance(50.0, 100.0)
    assert zone== "paracentral"

def test_zone_peripheral():
    zone, dn=clinical_zone_from_distance(90.0, 100.0)
    assert zone=="peripheral"

def test_zone_at_centre():
    zone, dn=clinical_zone_from_distance(0.0, 100.0)
    assert zone=="central"
    assert dn== 0.0

# Lesion Quadrant Tests

def test_quadrant_superior_nasal_right_eye():
    result = lesion_quadrant(cx=40, cy=30, ccx=50, ccy=50, eye="Right")
    assert result["vertical_sector"] == "superior"
    assert result["horizontal_sector"] == "nasal"

def test_quadrant_inferior_temporal_right_eye():
    result = lesion_quadrant(cx=60, cy=70, ccx=50, ccy=50, eye="Right")
    assert result["vertical_sector"] == "inferior"
    assert result["horizontal_sector"] == "temporal"

def test_quadrant_left_eye_flips_horizontal():
    # same position but left eye should flip nasal/temporal
    right = lesion_quadrant(cx=40, cy=30, ccx=50, ccy=50, eye="Right")
    left = lesion_quadrant(cx=40, cy=30, ccx=50, ccy=50, eye="Left")
    assert right["horizontal_sector"] != left["horizontal_sector"]

# Blur / variance of laplacian tests
def test_blur_flat_image_is_low():
    # completely flat image should have near zero laplacian variance
    flat = np.ones((100, 100), dtype=np.uint8) * 128
    score = variance_of_laplacian(flat)
    assert score < 1.0

def test_blur_noisy_image_is_higher():
    # random noise image should have higher laplacian variance than flat
    flat = np.ones((100, 100), dtype=np.uint8) * 128
    noisy = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    assert variance_of_laplacian(noisy) > variance_of_laplacian(flat)