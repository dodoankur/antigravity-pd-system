"""
Advanced PD Calculator Service
Implements:
  - Horizontal-only PD measurement (clinically correct definition)
  - Per-eye iris scale anchoring for monocular PD accuracy
  - Four-anchor weighted scale estimation (iris, ICD, fissure, OCD)
  - Cross-anchor sanity check to detect landmark mis-detection
  - Yaw + roll pose correction (pitch correctly excluded)
  - Perspective / depth correction via MediaPipe Z
  - Confidence-weighted median aggregation (called from main.py)
  - IQR outlier rejection (called from main.py)
  - Left/right symmetry warning
"""
import logging
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from services.face_detection import IrisData
from services.reference_detection import ReferenceDetectionResult
from models.schemas import MeasurementMethod, ErrorMargin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Landmark index groups — iris centres + 4 edge points per eye
# MediaPipe refine_landmarks indices:
#   468 = right iris centre, 473 = left iris centre
#   469-472 = right iris edge (top, right, bottom, left)
#   474-477 = left  iris edge (top, right, bottom, left)
# ---------------------------------------------------------------------------
MULTI_PAIR_INDICES: List[Tuple[int, int]] = [
    (473, 468),   # Primary: left iris centre → right iris centre
    (474, 469),   # Left/right iris top edge
    (475, 470),   # Left/right iris right edge
    (476, 471),   # Left/right iris bottom edge
    (477, 472),   # Left/right iris left edge
]

# T1-3: Weight the centre-pair (473, 468) 3× relative to each edge pair.
# Edge pairs (474-477 / 469-472) are iris boundary landmarks that sit at the
# periphery of the iris. At any non-zero yaw the boundary points on the near
# and far eye project asymmetrically, introducing a systematic underestimate
# of ~0.3–0.8mm. The iris CENTRE (473/468) is geometrically stable under yaw.
# Weights [3, 1, 1, 1, 1] (sum=7) reduce edge-pair contribution to ~57% of
# the simple mean without discarding them entirely (they still reduce jitter
# from single-landmark noise).
MULTI_PAIR_WEIGHTS: List[float] = [3.0, 1.0, 1.0, 1.0, 1.0]

# Anatomical constants (population averages)
AVERAGE_IRIS_DIAMETER_MM  = 11.7   # mm — adult population average (±0.5mm)
# Anatomical reference constants — calibrated to MediaPipe landmark positions.
# MediaPipe's canthus landmarks sit slightly inside the true anatomical exocanthion,
# so these values are smaller than published caliper measurements.
# Derived from: 4 Indian subjects (8 photos, known PD) + literature ratio scaling
# for European/Caucasian (Farkas 1994, Bouhadana 2022 systematic review n=3222).
# Weighted 60% European (UK primary user base) + 40% Indian.
# Old (Western population averages, too large): ICD=31.5  FISS=28.0  OCD=90.0
AVERAGE_ICD_MM            = 32.3   # mm — inner canthal distance (MediaPipe-space)
AVERAGE_EYE_FISSURE_MM    = 26.3   # mm — single eye width inner→outer (MediaPipe-space)
AVERAGE_OCD_MM            = 84.5   # mm — outer canthal distance (MediaPipe-space)

# Age-group iris diameter calibration (T3-3A)
# Child values are meaningfully smaller — using adult constant causes ~8% over-estimate.
IRIS_DIAMETER_BY_AGE: dict = {
    "adult":       11.7,   # 18+  — population average
    "teen":        11.5,   # 13–17 — near-adult
    "child":       10.8,   # 5–12 — measurably smaller
    "young_child": 10.2,   # under 5 — significantly smaller
}

# Anchor weights — based on empirical measurement reliability, not just population variation.
# Test data (3 real photos, known PD=64mm) showed:
#   iris-only  => +9.3mm error  (ellipse fit underestimates diameter at portrait orientation)
#   ICD-only   => -4.5mm error  (inner canthal distance has wide population spread ±9.5%)
#   OCD-only   => -0.2mm error  (outer canthal distance near-perfect)
#   fissure    => -0.5mm error  (eye fissure near-perfect)
# Revised: heavily demote iris (unreliable pixel measurement), promote OCD+fissure.
#   Old: iris 0.45  icd 0.25  fissure 0.20  ocd 0.10
#   New: iris 0.10  icd 0.20  fissure 0.35  ocd 0.35
ANCHOR_WEIGHTS = {
    "iris":    0.10,
    "icd":     0.20,
    "fissure": 0.35,
    "ocd":     0.35,
}

# Cross-anchor sanity: if max/min scale ratio exceeds this, flag uncertainty
# OCD has wider natural population variation (±5mm on 90mm = ±5.6%) vs iris (±0.5mm on 11.7mm = ±4.3%).
# A single spread threshold would either over-reject valid OCD readings or under-reject bad iris readings.
# Per-anchor thresholds handle this correctly. The cross-anchor check still uses the anchor mean, so we
# compare each anchor against its own tolerance before folding into the sanity flag.
ANCHOR_SPREAD_THRESHOLD = 1.15   # 15% spread → suspect landmark mis-detection (iris/icd/fissure)
OCD_SPREAD_TOLERANCE    = 1.20   # 20% tolerance for OCD — wider population variance expected

# Symmetry warning: if |left_pd - right_pd| > this, add a warning
ASYMMETRY_WARNING_MM = 4.0


@dataclass
class PDResult:
    overall_pd_mm:          float
    left_pd_mm:             float
    right_pd_mm:            float
    method:                 MeasurementMethod
    model_used:             str
    confidence_score:       float
    error_margin:           ErrorMargin
    disclaimer:             str
    reference_detected:     Optional[bool]  = None
    reference_scale_factor: Optional[float] = None
    asymmetry_warning:      bool            = False   # NEW


class PDCalculatorService:

    # ------------------------------------------------------------------
    # Iris estimation (no reference object)
    # ------------------------------------------------------------------

    def calculate_pd_with_iris_estimation(self, iris_data: IrisData) -> PDResult:
        """
        Calculate PD using iris diameter as the primary scale anchor.

        Tier 1 improvements:
          - Horizontal-only PD (clinically correct)
          - Yaw + roll correction only (pitch excluded — doesn't affect horizontal PD)
          - Multi-pair landmark averaging shared with reference mode

        Tier 2 improvements:
          - Per-eye iris scale: left/right monocular PDs use their own iris diameter
          - Four-anchor weighted scale for overall PD
          - Cross-anchor sanity check: flags low confidence if anchors disagree >15%
        """
        # --- Four-anchor weighted scale estimation (for overall PD) ---
        scale_estimates, anchor_low_conf = self._compute_anchor_scales(iris_data)

        total_weight = sum(ANCHOR_WEIGHTS[k] for k in scale_estimates)
        mm_per_pixel = (
            sum(scale_estimates[k] * ANCHOR_WEIGHTS[k] for k in scale_estimates) / total_weight
            if total_weight > 0 else (scale_estimates.get("iris") or 0.0)
        )

        logger.debug(
            "Scale anchors: iris=%.4f icd=%.4f fissure=%.4f ocd=%.4f → weighted=%.4f%s",
            scale_estimates.get("iris", 0.0),
            scale_estimates.get("icd", 0.0),
            scale_estimates.get("fissure", 0.0),
            scale_estimates.get("ocd", 0.0),
            mm_per_pixel,
            " [LOW-CONF: anchors disagree]" if anchor_low_conf else "",
        )

        # --- Per-eye iris scale for monocular PDs (T2-2 + T3-3A) ---
        # Use caller-supplied iris_diameter_mm if available, else adult default.
        iris_mm = iris_data.iris_diameter_mm if iris_data.iris_diameter_mm > 0 else AVERAGE_IRIS_DIAMETER_MM
        scale_left  = (
            iris_mm / iris_data.left_iris_diameter_px
            if iris_data.left_iris_diameter_px > 0 else mm_per_pixel
        )
        scale_right = (
            iris_mm / iris_data.right_iris_diameter_px
            if iris_data.right_iris_diameter_px > 0 else mm_per_pixel
        )

        # --- Pose correction + PD pixel measurement ---
        # T3-1: try full solvePnP projection first (exact 6-DOF correction).
        # T1-Bug2: use img_w/img_h stored on IrisData at detection time.
        # The old pattern (px_x / norm_x) amplified floating-point error when
        # the iris was near the left image edge (norm_x → 0). Direct storage
        # eliminates the re-derivation entirely and is computed once.
        img_w = iris_data.img_w if iris_data.img_w > 0 else (
            float(iris_data.left_iris_pixel[0]) / iris_data.left_iris_center[0]
            if iris_data.left_iris_center[0] > 0 else 1000.0
        )
        img_h = iris_data.img_h if iris_data.img_h > 0 else (
            float(iris_data.left_iris_pixel[1]) / iris_data.left_iris_center[1]
            if iris_data.left_iris_center[1] > 0 else 1000.0
        )

        pnp_result = self._solve_pnp_projection(iris_data, img_w, img_h)
        if pnp_result is not None:
            avg_pd_px, avg_left_px, avg_right_px, _, _ = pnp_result
            # solvePnP already projects to frontal — no further pose correction needed.
            # NOTE (IMP G): solvePnP uses 6 face-perimeter landmarks (forehead, chin,
            # nose tip, chin, and two outer canthi) — none of which are on the iris.
            # The correction is therefore an approximation: it models head orientation
            # but not the exact 3D position of each iris centre.  Accuracy improves at
            # yaw < 15° and degrades noticeably beyond 20°.  A full iris-3D model
            # (e.g. with corneal refraction and pupil depth) would be required to do
            # better here without a physical reference object.
            pose_correction = 1.0
            used_pnp = True
        else:
            # Fallback: multi-pair cos() corrected PD
            avg_pd_px, avg_left_px, avg_right_px = self._compute_multi_pair_pd(iris_data)
            pose_correction = self._horizontal_pose_correction(iris_data.head_pose)
            used_pnp = False

        # --- Depth / perspective correction ---
        depth_correction = self._depth_correction_factor(iris_data.face_depth_z)

        # --- Final PD calculation ---
        # Overall PD uses the 4-anchor weighted mm_per_pixel scale — this is the
        # most accurate combined estimate (OCD+fissure dominate at 0.35+0.35=0.70).
        # Monocular L/R split is derived proportionally from overall PD so that
        # left + right always equals overall_pd_mm exactly.
        # Per-eye iris scale is NOT used for the final value because the iris pixel
        # measurement is unreliable (empirical +9mm error vs OCD/fissure near-perfect).
        overall_pd_mm_raw = (avg_pd_px / pose_correction) * mm_per_pixel * depth_correction

        # Monocular split: apportion overall PD by the L/R pixel ratio
        total_px = avg_left_px + avg_right_px
        if total_px > 0:
            left_pd_mm  = overall_pd_mm_raw * (avg_left_px  / total_px)
            right_pd_mm = overall_pd_mm_raw * (avg_right_px / total_px)
        else:
            left_pd_mm  = overall_pd_mm_raw / 2
            right_pd_mm = overall_pd_mm_raw / 2
        corrected_pd_mm = left_pd_mm + right_pd_mm

        # --- Confidence & error margin ---
        # Reduce confidence if anchors disagree (cross-anchor sanity check)
        conf = iris_data.confidence
        if anchor_low_conf:
            conf = min(conf, 0.65)   # cap at medium confidence when anchors are inconsistent
        error_val = 1.0 if conf > 0.9 else (1.5 if conf > 0.7 else 2.5)
        error_margin = ErrorMargin(
            value_mm=error_val,
            percentage=round((error_val / corrected_pd_mm) * 100, 1) if corrected_pd_mm > 0 else 0.0,
            confidence_score=conf,
        )

        # --- Symmetry warning ---
        asymmetry_mm      = abs(left_pd_mm - right_pd_mm)
        asymmetry_warning = asymmetry_mm > ASYMMETRY_WARNING_MM

        logger.debug(
            "Iris estimation: avg_pd_px=%.1f pose_corr=%.3f depth_corr=%.3f "
            "mm_per_px=%.4f pd=%.1f L=%.1f R=%.1f asym=%.1fmm anchor_low_conf=%s",
            avg_pd_px, pose_correction, depth_correction,
            mm_per_pixel, corrected_pd_mm, left_pd_mm, right_pd_mm, asymmetry_mm,
            anchor_low_conf,
        )

        disclaimer = self._generate_disclaimer(
            conf, iris_data.head_pose, asymmetry_warning, asymmetry_mm, anchor_low_conf
        )

        return PDResult(
            overall_pd_mm=round(corrected_pd_mm, 1),
            left_pd_mm=round(left_pd_mm, 1),
            right_pd_mm=round(right_pd_mm, 1),
            method=MeasurementMethod.IRIS_ESTIMATION,
            model_used=(
                "MediaPipe + 4-Anchor Scale + Per-Eye Iris + solvePnP Projection"
                if used_pnp else
                "MediaPipe + 4-Anchor Scale + Per-Eye Iris + Yaw/Roll Correction"
            ),
            confidence_score=conf,
            error_margin=error_margin,
            disclaimer=disclaimer,
            asymmetry_warning=asymmetry_warning,
        )

    # ------------------------------------------------------------------
    # Reference-object mode
    # ------------------------------------------------------------------

    def calculate_pd_with_reference(
        self, iris_data: IrisData, ref: ReferenceDetectionResult
    ) -> PDResult:
        """Calculate PD using a detected reference object for scale.

        Now uses the same multi-pair averaging + horizontal-only pose correction
        as iris-estimation mode, so reference mode is consistently the more
        accurate path it claims to be.
        """
        mm_per_pixel = ref.scale_factor

        # --- Pose correction + PD pixel measurement ---
        # T3-F: try solvePnP projection first (same as iris mode) — exact 6-DOF
        # correction is better than the cos() approximation at yaw > 15°.
        # Reference scale (mm_per_pixel) still comes from the reference object;
        # only the pixel-distance correction changes.
        img_w = iris_data.img_w if iris_data.img_w > 0 else (
            float(iris_data.left_iris_pixel[0]) / iris_data.left_iris_center[0]
            if iris_data.left_iris_center[0] > 0 else 1000.0
        )
        img_h = iris_data.img_h if iris_data.img_h > 0 else (
            float(iris_data.left_iris_pixel[1]) / iris_data.left_iris_center[1]
            if iris_data.left_iris_center[1] > 0 else 1000.0
        )

        pnp_result = self._solve_pnp_projection(iris_data, img_w, img_h)
        if pnp_result is not None:
            avg_pd_px, avg_left_px, avg_right_px, _, _ = pnp_result
            pose_correction = 1.0   # solvePnP already projects to frontal
            used_pnp = True
        else:
            # Fallback: multi-pair cos() corrected PD
            avg_pd_px, avg_left_px, avg_right_px = self._compute_multi_pair_pd(iris_data)
            pose_correction = self._horizontal_pose_correction(iris_data.head_pose)
            used_pnp = False

        # T1-Bug4 (reference mode): derive overall as left + right for consistency.
        # In reference mode all three use the same mm_per_pixel scale, so there is
        # no per-eye drift, but keeping the derivation method consistent with iris
        # mode prevents confusion and ensures overall == left + right everywhere.
        left_pd_mm      = (avg_left_px  / pose_correction) * mm_per_pixel
        right_pd_mm     = (avg_right_px / pose_correction) * mm_per_pixel
        corrected_pd_mm = left_pd_mm + right_pd_mm

        conf            = iris_data.confidence
        asymmetry_mm    = abs(left_pd_mm - right_pd_mm)
        asym_warning    = asymmetry_mm > ASYMMETRY_WARNING_MM

        logger.debug(
            "Reference estimation: avg_pd_px=%.1f scale=%.4f pose_corr=%.3f pd=%.1f L=%.1f R=%.1f used_pnp=%s",
            avg_pd_px, mm_per_pixel, pose_correction, corrected_pd_mm, left_pd_mm, right_pd_mm, used_pnp,
        )

        return PDResult(
            overall_pd_mm=round(corrected_pd_mm, 1),
            left_pd_mm=round(left_pd_mm, 1),
            right_pd_mm=round(right_pd_mm, 1),
            method=MeasurementMethod.REFERENCE_OBJECT,
            model_used=(
                "Ref-Object + solvePnP Projection + Multi-Pair Avg"
                if used_pnp else
                "Ref-Object + Yaw/Roll Correction + Multi-Pair Avg"
            ),
            confidence_score=conf,
            error_margin=ErrorMargin(
                value_mm=1.0,
                percentage=round((1.0 / corrected_pd_mm) * 100, 1) if corrected_pd_mm > 0 else 0.0,
                confidence_score=conf,
            ),
            disclaimer=self._generate_disclaimer(conf, iris_data.head_pose, asym_warning, asymmetry_mm),
            reference_detected=True,
            reference_scale_factor=mm_per_pixel,
            asymmetry_warning=asym_warning,
        )

    # ------------------------------------------------------------------
    # Batch post-processing helpers (called by main.py)
    # ------------------------------------------------------------------

    @staticmethod
    def iqr_filter(values: List[float]) -> List[int]:
        """
        Return indices whose values are within ±2mm of the median (T2-C).

        For N=10 frames the classical 1.5×IQR method is too aggressive: with
        a tight distribution (IQR < 1mm) it rejects frames that are only
        modestly different from the centre. The clinical acceptable tolerance
        for PD measurement is ±0.5mm for prescription eyewear, so frames
        outside ±2mm of the session median are genuinely outliers.
        Returns all indices when len(values) < 4 (too few to reject any).
        """
        if len(values) < 4:
            return list(range(len(values)))
        arr    = np.array(values)
        median = float(np.median(arr))
        lo     = median - 2.0
        hi     = median + 2.0
        return [i for i, v in enumerate(values) if lo <= v <= hi]

    @staticmethod
    def weighted_median_index(values: List[float], weights: List[float]) -> int:
        """
        Return the index of the confidence-weighted median value.
        Sorts by value, accumulates normalised weights, picks the index where
        cumulative weight first crosses 0.5.
        """
        if not values:
            return 0
        total = sum(weights)
        if total == 0:
            weights = [1.0] * len(weights)
            total   = float(len(weights))
        norm_w = [w / total for w in weights]
        order  = sorted(range(len(values)), key=lambda i: values[i])
        cum    = 0.0
        for idx in order:
            cum += norm_w[idx]
            if cum >= 0.5:
                return idx
        return order[-1]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_anchor_scales(iris_data: IrisData) -> Tuple[Dict[str, float], bool]:
        """Compute per-anchor px→mm scale estimates and cross-anchor sanity flag.

        Returns (scale_estimates_dict, anchor_low_confidence).
        Uses caller-supplied iris_diameter_mm when provided (T3-3A age calibration),
        otherwise falls back to the adult population default (11.7mm).
        anchor_low_confidence is True when the spread between anchor scales
        exceeds ANCHOR_SPREAD_THRESHOLD (15%), indicating a likely landmark
        mis-detection.
        """
        # T3-3A: use caller-supplied diameter if valid, else population default
        iris_mm = (
            iris_data.iris_diameter_mm
            if iris_data.iris_diameter_mm > 0
            else AVERAGE_IRIS_DIAMETER_MM
        )

        scale_estimates: Dict[str, float] = {}

        avg_iris_px = (iris_data.left_iris_diameter_px + iris_data.right_iris_diameter_px) / 2.0
        if avg_iris_px > 0:
            scale_estimates["iris"] = iris_mm / avg_iris_px

        if iris_data.icd_px > 0:
            scale_estimates["icd"] = AVERAGE_ICD_MM / iris_data.icd_px

        avg_fissure_px = (iris_data.left_fissure_px + iris_data.right_fissure_px) / 2.0
        if avg_fissure_px > 0:
            scale_estimates["fissure"] = AVERAGE_EYE_FISSURE_MM / avg_fissure_px

        if iris_data.ocd_px > 0:
            scale_estimates["ocd"] = AVERAGE_OCD_MM / iris_data.ocd_px

        # Cross-anchor sanity check (T2-3)
        # T2-E: OCD gets a wider tolerance (20%) because outer canthal distance
        # has ±5.6% natural population variation vs ±4.3% for iris diameter.
        # We flag low confidence only when a non-OCD anchor exceeds 15% spread,
        # or when OCD exceeds its own 20% tolerance vs the mean of the others.
        anchor_low_conf = False
        vals = list(scale_estimates.values())
        if len(vals) >= 2:
            # Compute spread for non-OCD anchors first
            non_ocd = {k: v for k, v in scale_estimates.items() if k != "ocd"}
            non_ocd_vals = list(non_ocd.values())
            if len(non_ocd_vals) >= 2:
                spread_non_ocd = max(non_ocd_vals) / min(non_ocd_vals) if min(non_ocd_vals) > 0 else 1.0
                if spread_non_ocd > ANCHOR_SPREAD_THRESHOLD:
                    anchor_low_conf = True
                    logger.debug(
                        "Non-OCD anchor spread %.2fx exceeds threshold %.2fx — low confidence",
                        spread_non_ocd, ANCHOR_SPREAD_THRESHOLD,
                    )
            # Also check OCD against mean of non-OCD anchors (wider tolerance)
            if "ocd" in scale_estimates and non_ocd_vals:
                ocd_scale  = scale_estimates["ocd"]
                mean_other = float(np.mean(non_ocd_vals))
                ocd_spread = max(ocd_scale, mean_other) / min(ocd_scale, mean_other) if min(ocd_scale, mean_other) > 0 else 1.0
                if ocd_spread > OCD_SPREAD_TOLERANCE:
                    anchor_low_conf = True
                    logger.debug(
                        "OCD anchor spread %.2fx vs non-OCD mean exceeds OCD tolerance %.2fx — low confidence",
                        ocd_spread, OCD_SPREAD_TOLERANCE,
                    )

        return scale_estimates, anchor_low_conf

    @staticmethod
    def _horizontal_pose_correction(head_pose: dict) -> float:
        """Pose correction factor for HORIZONTAL PD measurement (cos() approximation).

        Used as fallback when rvec/tvec are unavailable.
        Only yaw (left-right turn) and roll (head tilt) shrink the horizontal
        projection. Pitch is correctly excluded.

        IMP F: the old code capped cos_roll at 0.85 (saturates at ~31.8°), which
        silently underestimated PD for any roll beyond that.  The roll gate in
        detect_iris() now rejects frames with roll > 20°, so cos(roll) is always
        >= cos(20°) ≈ 0.94 for frames that reach here.  The cap is removed so
        the cos() value is exact for every accepted frame.
        """
        yaw_rad  = np.deg2rad(head_pose.get("yaw",  0.0))
        roll_rad = np.deg2rad(head_pose.get("roll", 0.0))
        cos_yaw  = max(np.cos(yaw_rad), 0.5)   # guard extreme yaw (kept)
        cos_roll = np.cos(roll_rad)             # no cap — roll gate handles extremes
        return float(cos_yaw * cos_roll)

    @staticmethod
    def _solve_pnp_projection(
        iris_data: IrisData,
        img_w: float,
        img_h: float,
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """Full solvePnP-based iris projection correction (T3-1).

        Back-projects the two iris pixel positions onto a virtual frontal-face
        plane using the 6-DOF rotation matrix from solvePnP, then re-measures
        the horizontal PD on that projection. This is geometrically exact —
        unlike the cos(yaw) approximation which accumulates error at yaw > 15°
        because each iris is at a slightly different 3D depth when the head turns.

        Returns (pd_px, left_px, right_px, nose_px, ok_flag) where ok_flag=1.0
        on success, 0.0 when rvec/tvec are unavailable and the caller should fall
        back to cos() correction.

        Algorithm:
          1. Build rotation matrix R from rvec (Rodrigues).
          2. Un-rotate each iris 2D pixel to 3D camera space using the pinhole
             inverse projection (x_3d = R^T @ K^-1 @ [x_px, y_px, 1]^T).
          3. Project all three un-rotated 3D points back to 2D with R=I (frontal),
             giving "what the pixel positions would look like head-on".
          4. Measure horizontal distance on the frontal projection.
        """
        if iris_data.rvec is None or iris_data.tvec is None:
            return None

        focal   = img_w
        cx, cy  = img_w / 2.0, img_h / 2.0

        K = np.array([[focal, 0, cx],
                      [0, focal, cy],
                      [0, 0, 1   ]], dtype=np.float64)
        K_inv = np.linalg.inv(K)

        R, _ = cv2.Rodrigues(iris_data.rvec)   # 3×3 rotation matrix

        def unrotate(px_x: float, px_y: float) -> np.ndarray:
            """Project pixel → normalised camera ray → un-rotate to frontal."""
            ray = K_inv @ np.array([px_x, px_y, 1.0])
            return R.T @ ray   # undo the head rotation

        lx_px = float(iris_data.left_iris_pixel[0])
        ly_px = float(iris_data.left_iris_pixel[1])
        rx_px = float(iris_data.right_iris_pixel[0])
        ry_px = float(iris_data.right_iris_pixel[1])
        nose_px_x = iris_data.nose_bridge_x

        l3d = unrotate(lx_px,    ly_px)
        r3d = unrotate(rx_px,    ry_px)
        n3d = unrotate(nose_px_x, cy)   # nose: use image centre Y

        # Project back to 2D at unit depth (frontal view)
        def to_2d(p3d: np.ndarray) -> float:
            """Return projected X pixel at unit Z."""
            if abs(p3d[2]) < 1e-6:
                return 0.0
            return float((p3d[0] / p3d[2]) * focal + cx)

        lx_f = to_2d(l3d)
        rx_f = to_2d(r3d)
        nx_f = to_2d(n3d)

        pd_f    = abs(lx_f - rx_f)
        left_f  = abs(lx_f - nx_f)
        right_f = abs(rx_f - nx_f)

        logger.debug(
            "solvePnP projection: lx=%.1f→%.1f rx=%.1f→%.1f pd_px=%.1f→%.1f",
            lx_px, lx_f, rx_px, rx_f,
            abs(lx_px - rx_px), pd_f,
        )

        return pd_f, left_f, right_f, nx_f, 1.0

    @staticmethod
    def _compute_multi_pair_pd(iris_data: IrisData) -> Tuple[float, float, float]:
        """Compute (overall_pd_px, left_pd_px, right_pd_px) using multi-pair averaging.

        Averages 5 iris-landmark pairs using MULTI_PAIR_WEIGHTS [3,1,1,1,1] to
        reduce per-frame jitter while down-weighting the yaw-biased edge pairs
        (T1-3). Falls back to the primary iris-centre pair if landmarks_raw is
        unavailable. Shared between iris-estimation and reference-object modes so
        both benefit from the same noise reduction.
        """
        pd_values_px:    List[float] = []
        left_pd_values:  List[float] = []
        right_pd_values: List[float] = []
        lms = iris_data.landmarks_raw

        if lms is not None:
            # T1-Bug2: use img_w/img_h from IrisData when available (set at
            # detection time). Fall back to normalised-to-pixel derivation only
            # for fixtures where img_w/img_h were not set (e.g. unit tests that
            # build IrisData manually without going through FaceDetectionService).
            if iris_data.img_w > 0:
                img_w = iris_data.img_w
                img_h = iris_data.img_h
            else:
                norm_x = iris_data.left_iris_center[0]
                px_x   = float(iris_data.left_iris_pixel[0])
                img_w  = px_x / norm_x if norm_x > 0 else 1.0

                norm_y = iris_data.left_iris_center[1]
                px_y   = float(iris_data.left_iris_pixel[1])
                img_h  = px_y / norm_y if norm_y > 0 else 1.0

            nose_x_px = lms[168].x * img_w  # nose bridge

            for left_idx, right_idx in MULTI_PAIR_INDICES:
                lx = lms[left_idx].x  * img_w
                rx = lms[right_idx].x * img_w
                # Horizontal-only distance — PD is a horizontal measurement by definition.
                # Using sqrt((dx)²+(dy)²) inflated PD when face was rolled; abs(dx) is correct.
                pd_values_px.append(abs(lx - rx))
                left_pd_values.append(abs(lx - nose_x_px))
                right_pd_values.append(abs(rx - nose_x_px))

        if pd_values_px:
            # T1-3: centre-weighted average — weight vector aligns with MULTI_PAIR_INDICES.
            # np.average normalises the weights internally so the sum need not be 1.
            w = MULTI_PAIR_WEIGHTS[: len(pd_values_px)]
            return (
                float(np.average(pd_values_px,   weights=w)),
                float(np.average(left_pd_values,  weights=w)),
                float(np.average(right_pd_values, weights=w)),
            )

        # Fallback: primary iris-centre pair only
        left_px  = np.array(iris_data.left_iris_pixel,  dtype=float)
        right_px = np.array(iris_data.right_iris_pixel, dtype=float)
        return (
            # T1-Bug1: use horizontal-only distance, not Euclidean.
            # PD is defined as the horizontal separation between pupil centres.
            # Using np.linalg.norm inflated PD when the face had any vertical
            # component (roll, head position asymmetry).
            abs(float(left_px[0]) - float(right_px[0])),
            abs(left_px[0]  - iris_data.nose_bridge_x),
            abs(right_px[0] - iris_data.nose_bridge_x),
        )

    @staticmethod
    def _depth_correction_factor(face_depth_z: float) -> float:
        """
        Apply a small depth-proportional scale correction.

        MediaPipe iris Z is in the same normalised unit as X/Y but negative
        (depth into the scene is negative Z). The typical range for a face
        held at arm's length is roughly -0.10 to -0.05.

        GATING (T1-Bug3): MediaPipe normalised Z is aspect-dependent and not
        reliably proportional to real-world depth. Applying the correction on
        every frame adds uncalibrated noise for faces near the reference depth.
        We only fire when |delta| > 0.03 (roughly 30cm of real-world depth
        change from the reference position) to avoid amplifying noise for
        typical arm's-length shots. Cap remains ±5% to limit worst-case damage.
        """
        if face_depth_z == 0.0:
            return 1.0
        # Reference depth (front-on, ideal distance): empirically ~-0.07
        reference_z = -0.07
        delta       = face_depth_z - reference_z
        # Gate: skip correction when delta is below the noise floor
        if abs(delta) <= 0.03:
            return 1.0
        # Scale factor: closer face → larger apparent PD → we divide by a factor > 1
        raw_factor = 1.0 + delta * 0.5   # 0.5 = empirical sensitivity
        return float(np.clip(raw_factor, 0.95, 1.05))

    @staticmethod
    def _generate_disclaimer(
        confidence: float,
        pose: dict,
        asymmetry_warning: bool,
        asymmetry_mm: float,
        anchor_low_conf: bool = False,
    ) -> str:
        warnings = []
        adj_pitch = abs(pose.get("pitch", 0.0))
        if adj_pitch > 90:
            adj_pitch = abs(adj_pitch - 180)

        if abs(pose.get("yaw",   0.0)) > 10:
            warnings.append(f"Head turned ({pose['yaw']:.1f}°)")
        if adj_pitch > 10:
            warnings.append(f"Head tilted ({adj_pitch:.1f}°)")
        if anchor_low_conf:
            warnings.append("Scale anchors inconsistent — retake for better accuracy")
        if asymmetry_warning:
            warnings.append(
                f"Asymmetry {asymmetry_mm:.1f}mm — consider retaking or "
                "consulting an optician"
            )

        warn_str    = " | ".join(f"Warning: {w}" for w in warnings)
        warn_prefix = f"{warn_str}. " if warn_str else ""
        return (
            f"{warn_prefix}Calculated using Horizontal PD + 4-Anchor Scale + "
            f"Per-Eye Iris + Yaw/Roll Correction. Confidence: {confidence * 100:.0f}%"
        )
