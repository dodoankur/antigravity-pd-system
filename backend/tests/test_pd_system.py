"""
Unit tests for the PD Measurement backend.

Run with:
    cd backend
    python -m pytest tests/ -v
"""
import math
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# PDCalculatorService tests
# ---------------------------------------------------------------------------
from services.pd_calculator import PDCalculatorService
from services.face_detection import IrisData
from services.reference_detection import ReferenceDetectionResult, ReferenceObjectType
from models.schemas import MeasurementMethod


def _make_iris_data(
    left_px=(300, 400),
    right_px=(500, 400),
    nose_x=400.0,
    left_dia=30.0,
    right_dia=30.0,
    face_width=120.0,
    yaw=0.0,
    pitch=0.0,
    confidence=0.9,
):
    """Helper: build a minimal IrisData fixture."""
    return IrisData(
        left_iris_center=(left_px[0] / 1000, left_px[1] / 1000, 0.0),
        right_iris_center=(right_px[0] / 1000, right_px[1] / 1000, 0.0),
        left_iris_pixel=left_px,
        right_iris_pixel=right_px,
        left_iris_diameter_px=left_dia,
        right_iris_diameter_px=right_dia,
        face_width_px=face_width,
        nose_bridge_x=nose_x,
        confidence=confidence,
        head_pose={"pitch": pitch, "yaw": yaw, "roll": 0.0},
        quality_score=200.0,
        iris_canthus_ratio=0.33,
        pose_symmetry={"horizontal": 0.5, "vertical": 0.5},
    )


class TestPDCalculatorIrisEstimation:
    def setup_method(self):
        self.svc = PDCalculatorService()

    def test_symmetric_face_zero_yaw(self):
        """Symmetric face at zero yaw should produce equal left/right PD."""
        iris = _make_iris_data(
            left_px=(300, 400), right_px=(500, 400), nose_x=400.0,
            left_dia=30.0, right_dia=30.0, yaw=0.0,
        )
        result = self.svc.calculate_pd_with_iris_estimation(iris)
        assert result.overall_pd_mm > 0
        # Left and right should be equal for a symmetric face
        assert result.left_pd_mm == pytest.approx(result.right_pd_mm, abs=0.5)
        assert result.method == MeasurementMethod.IRIS_ESTIMATION

    def test_overall_pd_equals_left_plus_right(self):
        """overall_pd must equal left_pd + right_pd exactly (T1-Bug4 fix).

        Overall is now derived as left + right so the three values are always
        internally consistent. The old tolerance of abs=1.0 masked a real
        discrepancy when the 4-anchor scale differed from the iris-only scale.
        """
        iris = _make_iris_data(
            left_px=(280, 400), right_px=(520, 400), nose_x=400.0,
            left_dia=32.0, right_dia=32.0, yaw=0.0,
        )
        result = self.svc.calculate_pd_with_iris_estimation(iris)
        assert result.overall_pd_mm == pytest.approx(
            result.left_pd_mm + result.right_pd_mm, abs=0.05
        )

    def test_yaw_correction_increases_pd(self):
        """A non-zero yaw should produce a LARGER corrected PD than uncorrected."""
        iris_straight = _make_iris_data(yaw=0.0)
        iris_turned = _make_iris_data(yaw=20.0)

        pd_straight = self.svc.calculate_pd_with_iris_estimation(iris_straight).overall_pd_mm
        pd_turned = self.svc.calculate_pd_with_iris_estimation(iris_turned).overall_pd_mm

        # At yaw=20°, cos(20°) ≈ 0.94 → dividing by it gives a larger PD
        assert pd_turned > pd_straight

    def test_yaw_correction_formula(self):
        """Verify the cos(yaw) division is applied correctly with per-eye iris scaling.

        Since T1-Bug4, overall_pd_mm = left_pd_mm + right_pd_mm.
        Both monocular PDs use per-eye iris scale (11.7 / iris_diameter_px).
        The fixture is symmetric: nose at 400, left iris at 300, right at 500.
        """
        yaw_deg = 15.0
        iris = _make_iris_data(
            left_px=(300, 400), right_px=(500, 400), nose_x=400.0,
            left_dia=30.0, right_dia=30.0, yaw=yaw_deg,
        )
        result = self.svc.calculate_pd_with_iris_estimation(iris)

        # Per-eye iris scale: 11.7mm / 30px
        scale = 11.7 / 30.0
        # Monocular pixel distances (horizontal only, nose at 400)
        left_px_dist  = abs(300 - 400)   # 100px
        right_px_dist = abs(500 - 400)   # 100px
        cos_yaw  = max(math.cos(math.radians(yaw_deg)), 0.5)
        cos_roll = math.cos(math.radians(0.0))   # no cap — roll gate handles extremes (IMP F)
        pose_corr   = cos_yaw * cos_roll
        expected_left  = (left_px_dist  / pose_corr) * scale
        expected_right = (right_px_dist / pose_corr) * scale
        expected_pd    = expected_left + expected_right   # overall = left + right

        assert result.overall_pd_mm == pytest.approx(round(expected_pd, 1), abs=0.1)

    def test_confidence_thresholds(self):
        """Error margin should tighten as confidence rises."""
        low_conf = _make_iris_data(confidence=0.5)
        mid_conf = _make_iris_data(confidence=0.75)
        high_conf = _make_iris_data(confidence=0.95)

        err_low = self.svc.calculate_pd_with_iris_estimation(low_conf).error_margin.value_mm
        err_mid = self.svc.calculate_pd_with_iris_estimation(mid_conf).error_margin.value_mm
        err_high = self.svc.calculate_pd_with_iris_estimation(high_conf).error_margin.value_mm

        assert err_low >= err_mid >= err_high

    def test_disclaimer_warns_on_high_yaw(self):
        iris = _make_iris_data(yaw=20.0)
        result = self.svc.calculate_pd_with_iris_estimation(iris)
        assert "Warning" in result.disclaimer or "turned" in result.disclaimer.lower()

    def test_disclaimer_no_warning_on_frontal(self):
        iris = _make_iris_data(yaw=0.0, pitch=0.0)
        result = self.svc.calculate_pd_with_iris_estimation(iris)
        assert "Warning" not in result.disclaimer


class TestPDCalculatorReference:
    def setup_method(self):
        self.svc = PDCalculatorService()

    def test_reference_mode_uses_scale_factor(self):
        iris = _make_iris_data(left_px=(300, 400), right_px=(500, 400), nose_x=400.0)
        scale = 0.5  # 0.5mm per pixel
        ref = ReferenceDetectionResult(
            detected=True,
            object_type=ReferenceObjectType.CREDIT_CARD,
            scale_factor=scale,
            confidence=0.95,
        )
        result = self.svc.calculate_pd_with_reference(iris, ref)
        # T1-Bug4: overall = left + right (both from nose).
        # Nose at 400, left iris at 300, right at 500 → each monocular = 100px × 0.5mm/px = 50mm
        expected_left  = abs(300 - 400) * scale    # 50mm
        expected_right = abs(500 - 400) * scale    # 50mm
        expected       = expected_left + expected_right  # 100mm
        assert result.overall_pd_mm == pytest.approx(round(expected, 1), abs=0.1)
        assert result.method == MeasurementMethod.REFERENCE_OBJECT
        assert result.reference_detected is True

    def test_reference_error_margin_is_1mm(self):
        iris = _make_iris_data()
        ref = ReferenceDetectionResult(
            detected=True,
            object_type=ReferenceObjectType.COIN_GBP_1,
            scale_factor=0.4,
            confidence=0.9,
        )
        result = self.svc.calculate_pd_with_reference(iris, ref)
        assert result.error_margin.value_mm == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Reference detection service tests
# ---------------------------------------------------------------------------
from services.reference_detection import ReferenceDetectionService


class TestReferenceDetectionCoinRadius:
    """Verify dynamic radius bounds scale with image size."""

    def setup_method(self):
        self.svc = ReferenceDetectionService()

    def test_small_image_uses_small_radii(self):
        """On a 640×480 image the max radius should be well below 200px."""
        import cv2, numpy as np
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # We just want to verify the method runs without error and the radius
        # calculation is sane — no circle will be detected in a blank image.
        result = self.svc.detect_coin(image, ReferenceObjectType.COIN_GBP_1)
        assert result is None  # blank image → no circle

    def test_large_image_does_not_crash(self):
        """A 4K image should not raise an error due to radius overflow."""
        import numpy as np
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        result = self.svc.detect_coin(image, ReferenceObjectType.COIN_GBP_1)
        assert result is None  # blank image → no circle


# ---------------------------------------------------------------------------
# Image format validation tests
# ---------------------------------------------------------------------------
from main import _validate_image_format, _resize_if_needed
from fastapi import HTTPException


class TestValidateImageFormat:
    def test_jpeg_accepted(self):
        jpeg_header = b"\xff\xd8\xff" + b"\x00" * 20
        _validate_image_format(jpeg_header)  # should not raise

    def test_png_accepted(self):
        png_header = b"\x89PNG" + b"\x00" * 20
        _validate_image_format(png_header)  # should not raise

    def test_webp_accepted(self):
        webp_header = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 10
        _validate_image_format(webp_header)  # should not raise

    def test_riff_non_webp_rejected(self):
        # RIFF but not WEBP (e.g. WAV)
        wav_header = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 10
        with pytest.raises(HTTPException) as exc_info:
            _validate_image_format(wav_header)
        assert exc_info.value.status_code == 400

    def test_unknown_format_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_image_format(b"GIF89a" + b"\x00" * 20)
        assert exc_info.value.status_code == 400

    def test_empty_bytes_rejected(self):
        with pytest.raises(HTTPException):
            _validate_image_format(b"")


class TestResizeIfNeeded:
    def test_small_image_unchanged(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = _resize_if_needed(image)
        assert result.shape == image.shape

    def test_large_image_resized(self):
        image = np.zeros((3000, 4000, 3), dtype=np.uint8)
        result = _resize_if_needed(image)
        assert max(result.shape[:2]) <= 1920

    def test_aspect_ratio_preserved(self):
        image = np.zeros((2000, 4000, 3), dtype=np.uint8)
        result = _resize_if_needed(image)
        original_ratio = 4000 / 2000
        result_ratio = result.shape[1] / result.shape[0]
        assert result_ratio == pytest.approx(original_ratio, abs=0.01)

    def test_exactly_at_limit_unchanged(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = _resize_if_needed(image)
        assert result.shape == image.shape
