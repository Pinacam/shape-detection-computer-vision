"""
Tests for the geometric feature-extraction logic in src/shape_detector.py
and a sanity check on the labeled training data in data/shape_database.csv.

extract_features() is pure (contour in, feature vector out) and doesn't
touch the camera or the trained model, so it's fully testable headless.
"""
import csv
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shape_detector import MIN_AREA, extract_features  # noqa: E402

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "shape_database.csv")


def _contour_from_mask(draw_fn):
    """Render a shape into a blank mask and return its outer contour."""
    img = np.zeros((300, 300), dtype=np.uint8)
    draw_fn(img)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1
    return contours[0]


def test_small_contour_is_filtered_out_by_min_area():
    tiny = np.array([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]], dtype=np.int32)
    assert extract_features(tiny) is None


def test_circle_contour_is_highly_circular_and_convex():
    contour = _contour_from_mask(lambda img: cv2.circle(img, (150, 150), 80, 255, -1))
    result = extract_features(contour)
    assert result is not None

    features, (x, y, w, h) = result
    area, perimeter, *_, circularity, compactness, convexity = features

    assert area > MIN_AREA
    # a perfect circle has circularity == 1; allow for pixelation error
    assert 0.85 <= circularity <= 1.05
    # a circle's contour is already convex
    assert convexity >= 0.95
    # bounding box should be roughly square for a circle
    assert abs(w - h) <= 2


def test_square_contour_bounding_box_matches_shape():
    contour = _contour_from_mask(lambda img: cv2.rectangle(img, (75, 75), (225, 225), 255, -1))
    result = extract_features(contour)
    assert result is not None

    features, (x, y, w, h) = result
    area = features[0]

    assert w == pytest.approx(h, abs=2)
    # area of a filled square should roughly match its bounding box area
    assert area == pytest.approx(w * h, rel=0.05)
    # a square is convex, so convexity should be close to 1
    convexity = features[8]
    assert convexity >= 0.95


def test_feature_vector_shape_and_order():
    contour = _contour_from_mask(lambda img: cv2.circle(img, (150, 150), 60, 255, -1))
    features, bbox = extract_features(contour)
    assert len(features) == 9
    assert len(bbox) == 4


def test_shape_database_csv_has_expected_schema_and_labels():
    # utf-8-sig strips a leading BOM if present, so the first column name
    # comes through as "id" rather than "﻿id"
    with open(DATA_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 500  # ~600 labeled measurements per the README

    expected_columns = {
        "id", "label", "area", "perimeter",
        "bounding_x", "bounding_y", "bounding_width", "bounding_height",
        "circularity", "compactness", "convexity",
    }
    assert expected_columns <= set(rows[0].keys())

    labels = {row["label"] for row in rows}
    expected_labels = {
        "Bolt", "Diode", "Ferrite Bead", "Knob",
        "Nut", "Resistor", "Spring", "Washer", "Wire",
    }
    assert labels == expected_labels
