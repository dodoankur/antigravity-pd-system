"""
Advanced Face Detection Service using MediaPipe Face Mesh
Includes:
  - Full 3D head-pose estimation (pitch, yaw, roll)
  - Frame quality gating (blur, brightness, blink, yaw)
  - Multiple landmark pairs for robust iris centre estimation
  - IrisData.is_usable flag so callers can skip bad frames
"""
import cv2
import logging
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class IrisData:
    """Data structure for iris detection results"""
    left_iris_center: Tuple[float, float, float]
    right_iris_center: Tuple[float, float, float]
    left_iris_pixel: Tuple[int, int]
    right_iris_pixel: Tuple[int, int]
    left_iris_diameter_px: float
    right_iris_diameter_px: float
    face_width_px: float
    nose_bridge_x: float
    confidence: float
    # Advanced metadata
    head_pose: Dict[str, float]   # pitch, yaw, roll (degrees)
    quality_score: float          # Laplacian blur variance
    iris_canthus_ratio: float     # for self-calibration
    pose_symmetry: Dict[str, float]  # horizontal/vertical symmetry

    # --- Quality gate fields ---
    is_usable: bool = True
    reject_reason: str = ""
    landmarks_raw: Any = field(default=None, repr=False)
    face_depth_z: float = 0.0
    left_eye_openness: float = 1.0
    right_eye_openness: float = 1.0

    # --- Multi-anchor scale fields (pixels) ---
    # icd_px   : inner canthal distance (right inner → left inner)
    # ocd_px   : outer canthal distance (right outer → left outer)
    # left_fissure_px  : left eye width  (left inner → left outer canthus)
    # right_fissure_px : right eye width (right inner → right outer canthus)
    icd_px:           float = 0.0
    ocd_px:           float = 0.0
    left_fissure_px:  float = 0.0
    right_fissure_px: float = 0.0

    # --- Calibration (T3-3A / T3-3B) ---
    # Caller-supplied iris diameter in mm (from age-group selector or auto-detection).
    # 0.0 means "use population default" (11.7mm adult).
    iris_diameter_mm: float = 0.0

    # Auto-detected age group from facial proportions (T3-3B).
    # One of: "adult", "teen", "child", "young_child"
    # Default is "adult" so IRIS_DIAMETER_BY_AGE lookup never needs a fallback key.
    estimated_age_group: str = "adult"

    # --- solvePnP pose vectors (T3-1) ---
    # rvec/tvec from cv2.solvePnP — used by pd_calculator for full 6-DOF
    # projection correction instead of cos() approximation.
    # None when solvePnP failed (pd_calculator falls back to cos() correction).
    rvec: Optional[np.ndarray] = field(default=None, repr=False)
    tvec: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Image dimensions (T1-Bug2) ---
    # Stored directly at detection time to avoid fragile re-derivation in
    # pd_calculator (px_x / norm_x amplifies floating-point error when the
    # iris is near the image edge where norm_x is very small).
    img_w: float = 0.0
    img_h: float = 0.0


class FaceDetectionService:
    """
    Advanced Service for high-precision PD measurement features:
    1. MediaPipe Iris Refinement (refine_landmarks=True)
    2. Head Pose / Perspective Correction (pitch, yaw, roll)
    3. Image Quality Assessment (Blur / Lighting)
    4. Frame Gating (blink, yaw, blur, brightness)
    5. Multiple Canthus Landmark Pairs for cross-validation
    """

    # --- Landmark Indices ---
    RIGHT_IRIS_CENTER = 468
    LEFT_IRIS_CENTER  = 473
    RIGHT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]
    LEFT_IRIS_LANDMARKS  = [473, 474, 475, 476, 477]

    # Canthus (Eye Corners)
    LEFT_EYE_INNER  = 463
    LEFT_EYE_OUTER  = 263
    RIGHT_EYE_INNER = 33
    RIGHT_EYE_OUTER = 133

    # Additional canthus pairs for multi-pair averaging
    # (inner canthus alt, outer canthus alt)
    LEFT_EYE_INNER_ALT  = 362
    LEFT_EYE_OUTER_ALT  = 382
    RIGHT_EYE_INNER_ALT = 133
    RIGHT_EYE_OUTER_ALT = 153

    # ICD landmarks: the two inner eye corners (closest to nose bridge)
    # RIGHT inner = 133, LEFT inner = 362
    ICD_RIGHT = 133
    ICD_LEFT  = 362

    # OCD landmarks: the two outer eye corners (closest to temples)
    # RIGHT outer = 33, LEFT outer = 263
    OCD_RIGHT = 33
    OCD_LEFT  = 263

    # Eye fissure: inner→outer of each eye individually
    # Left eye:  inner=362, outer=263
    # Right eye: inner=133, outer=33
    LEFT_FISSURE_INNER  = 362
    LEFT_FISSURE_OUTER  = 263
    RIGHT_FISSURE_INNER = 133
    RIGHT_FISSURE_OUTER = 33

    # Eye-openness landmarks (upper / lower lid for each eye)
    LEFT_LID_UPPER  = 159
    LEFT_LID_LOWER  = 145
    RIGHT_LID_UPPER = 386
    RIGHT_LID_LOWER = 374

    # Blink threshold: openness ratio below this is considered a blink
    # 0.008 = eyes genuinely shut; 0.012 was too aggressive (heavy lids fail)
    BLINK_THRESHOLD = 0.008

    # Quality gates
    MAX_YAW_DEG   = 25.0   # degrees — beyond this PD is unreliable (outer check in main.py is 35°)
    MAX_PITCH_DEG = 20.0   # degrees — pitch matters less for horizontal PD
    MAX_ROLL_DEG  = 20.0   # degrees — beyond 20° the iris ellipse becomes significantly oblate;
                           # cos(20°)≈0.94 means up to 6% scale error even after correction.
                           # IMP F: gate instead of capping cos_roll at 0.85 (31.8°) which
                           # produced a fixed underestimate for any roll > 31.8°.
    MIN_BLUR        = 80.0  # Laplacian variance for UPLOAD photos (T2-2).
                            # Blurry iris edges inflate diameter estimates by 0.5–1mm,
                            # so we reject anything below this for static photos.
    MIN_BLUR_CAMERA = 35.0  # Laplacian variance for CAMERA / batch frames.
                            # Real Safari camera data shows frames scoring 40–60.
                            # 55 sat in the middle of that range and randomly rejected
                            # good frames. 35 accepts all real camera frames while
                            # still rejecting genuinely blurry ones (closed eyes,
                            # severe motion blur) which score <20.
    MIN_BRIGHTNESS = 35.0
    MAX_BRIGHTNESS = 230.0

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_quality(self, image: np.ndarray, camera_mode: bool = False) -> Dict[str, float]:
        """Calculates brightness and blur scores for the image.

        Args:
            image:       BGR image as numpy array.
            camera_mode: When True uses MIN_BLUR_CAMERA (55) instead of MIN_BLUR (80).
                         Camera frames from Safari/MediaPipe→canvas→JPEG are inherently
                         softer than static upload photos and need a lower threshold.
        """
        gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score  = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness  = float(np.mean(gray.astype(np.float32)))
        min_blur    = self.MIN_BLUR_CAMERA if camera_mode else self.MIN_BLUR
        is_reliable = (
            blur_score > min_blur
            and self.MIN_BRIGHTNESS < brightness < self.MAX_BRIGHTNESS
        )
        return {
            "blur":        round(blur_score, 2),
            "brightness":  round(brightness, 2),
            "is_reliable": is_reliable,
        }

    def detect_iris(self, image: np.ndarray, camera_mode: bool = False) -> Optional["IrisData"]:
        """
        Detect iris landmarks and compute all metadata.

        Returns an IrisData object where `is_usable=False` (with a
        `reject_reason`) when the frame fails a quality gate — the caller
        should skip/discard such frames rather than raising an exception,
        which would waste a slot in a batch.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        quality = self.assess_quality(image, camera_mode=camera_mode)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        pose, rvec, tvec = self.estimate_head_pose(landmarks, width, height)

        # --- Quality gates (return unusable IrisData instead of None so
        #     batch callers can count & log rejections) ---

        # Normalise pitch to (-90, 90)
        adj_pitch = abs(pose["pitch"])
        if adj_pitch > 90:
            adj_pitch = abs(adj_pitch - 180)

        if abs(pose["yaw"]) > self.MAX_YAW_DEG:
            return self._reject(
                landmarks, width, height, pose, quality,
                f"yaw={pose['yaw']:.1f}° exceeds {self.MAX_YAW_DEG}°",
            )
        if adj_pitch > self.MAX_PITCH_DEG:
            return self._reject(
                landmarks, width, height, pose, quality,
                f"pitch={adj_pitch:.1f}° exceeds {self.MAX_PITCH_DEG}°",
            )
        if abs(pose["roll"]) > self.MAX_ROLL_DEG:
            # IMP F: reject frames with excessive roll rather than capping
            # cos_roll at 0.85 (which saturated and hid a fixed underestimate
            # for any roll > 31.8°).  20° gives cos(20°)≈0.94 — within the
            # range where the cos() correction is still accurate.
            return self._reject(
                landmarks, width, height, pose, quality,
                f"roll={pose['roll']:.1f}° exceeds {self.MAX_ROLL_DEG}°",
            )
        if not quality["is_reliable"]:
            return self._reject(
                landmarks, width, height, pose, quality,
                f"blur={quality['blur']:.0f} brightness={quality['brightness']:.0f}",
            )

        # Blink detection
        left_openness  = self._eye_openness(landmarks, self.LEFT_LID_UPPER,  self.LEFT_LID_LOWER,  height)
        right_openness = self._eye_openness(landmarks, self.RIGHT_LID_UPPER, self.RIGHT_LID_LOWER, height)
        if left_openness < self.BLINK_THRESHOLD or right_openness < self.BLINK_THRESHOLD:
            return self._reject(
                landmarks, width, height, pose, quality,
                f"blink detected L={left_openness:.4f} R={right_openness:.4f}",
            )
        # --- Iris detection (T3-2: sub-pixel ellipse centre) ---
        nose_bridge   = landmarks[168]
        nose_bridge_x = nose_bridge.x * width

        # Fit ellipse to iris landmarks for sub-pixel centre + stable diameter
        l_cx, l_cy, left_diameter,  l_ok = self._ellipse_iris_centre(landmarks, self.LEFT_IRIS_LANDMARKS,  width, height)
        r_cx, r_cy, right_diameter, r_ok = self._ellipse_iris_centre(landmarks, self.RIGHT_IRIS_LANDMARKS, width, height)

        left_iris_px  = (int(round(l_cx)), int(round(l_cy)))
        right_iris_px = (int(round(r_cx)), int(round(r_cy)))

        # Use ellipse Z from MediaPipe centre landmark for depth
        left_iris_z  = landmarks[self.LEFT_IRIS_CENTER].z
        right_iris_z = landmarks[self.RIGHT_IRIS_CENTER].z

        # Normalised coords for IrisData (needed by pd_calculator multi-pair logic)
        left_iris_center_norm  = (l_cx / width, l_cy / height, left_iris_z)
        right_iris_center_norm = (r_cx / width, r_cy / height, right_iris_z)

        logger.debug(
            "Ellipse fit: L_ok=%s cx=%.2f cy=%.2f diam=%.2f | R_ok=%s cx=%.2f cy=%.2f diam=%.2f",
            bool(l_ok), l_cx, l_cy, left_diameter, bool(r_ok), r_cx, r_cy, right_diameter,
        )

        # Canthus cross-validation (primary pair)
        li = landmarks[self.LEFT_EYE_INNER]
        lo = landmarks[self.LEFT_EYE_OUTER]
        canthus_dist_px = float(np.sqrt(((li.x - lo.x) * width) ** 2 + ((li.y - lo.y) * height) ** 2))
        iris_canthus_ratio = (left_diameter / canthus_dist_px) if canthus_dist_px > 0 else 0.0

        pose_symmetry = self._calculate_pose_symmetry(landmarks)
        confidence    = self._calculate_advanced_confidence(
            pose, quality, left_diameter, right_diameter,
            ellipse_ok=l_ok > 0 or r_ok > 0,
            pnp_ok=rvec is not None,
            camera_mode=camera_mode,
        )
        estimated_age = self.estimate_age_group(landmarks, width, height)

        # Depth: average Z of the two iris centres (MediaPipe normalised)
        face_depth_z = float((left_iris_z + right_iris_z) / 2.0)

        # --- Multi-anchor scale measurements (horizontal-only — IMP D) ---
        # ICD, OCD, and fissure are anatomically horizontal spans; using
        # _landmark_horiz_px (abs dx) instead of Euclidean eliminates the
        # small inflation from roll / vertical landmark noise.
        icd_px           = self._landmark_horiz_px(landmarks, self.ICD_RIGHT,           self.ICD_LEFT,           width)
        ocd_px           = self._landmark_horiz_px(landmarks, self.OCD_RIGHT,           self.OCD_LEFT,           width)
        left_fissure_px  = self._landmark_horiz_px(landmarks, self.LEFT_FISSURE_INNER,  self.LEFT_FISSURE_OUTER, width)
        right_fissure_px = self._landmark_horiz_px(landmarks, self.RIGHT_FISSURE_INNER, self.RIGHT_FISSURE_OUTER,width)

        logger.debug(
            "Frame accepted: yaw=%.1f° pitch=%.1f° blur=%.0f bright=%.0f "
            "blink_L=%.4f blink_R=%.4f conf=%.3f | "
            "anchors: iris_L=%.1f iris_R=%.1f icd=%.1f ocd=%.1f fiss_L=%.1f fiss_R=%.1f",
            pose["yaw"], adj_pitch, quality["blur"], quality["brightness"],
            left_openness, right_openness, confidence,
            left_diameter, right_diameter, icd_px, ocd_px, left_fissure_px, right_fissure_px,
        )

        return IrisData(
            left_iris_center=left_iris_center_norm,
            right_iris_center=right_iris_center_norm,
            left_iris_pixel=left_iris_px,
            right_iris_pixel=right_iris_px,
            left_iris_diameter_px=left_diameter,
            right_iris_diameter_px=right_diameter,
            face_width_px=canthus_dist_px,
            nose_bridge_x=nose_bridge_x,
            confidence=confidence,
            head_pose=pose,
            quality_score=quality["blur"],
            iris_canthus_ratio=iris_canthus_ratio,
            pose_symmetry=pose_symmetry,
            is_usable=True,
            reject_reason="",
            landmarks_raw=landmarks,
            face_depth_z=face_depth_z,
            left_eye_openness=left_openness,
            right_eye_openness=right_openness,
            icd_px=icd_px,
            ocd_px=ocd_px,
            left_fissure_px=left_fissure_px,
            right_fissure_px=right_fissure_px,
            rvec=rvec,
            tvec=tvec,
            estimated_age_group=estimated_age,
            img_w=float(width),
            img_h=float(height),
        )

    # ------------------------------------------------------------------
    # Head Pose
    # ------------------------------------------------------------------

    def estimate_head_pose(
        self, landmarks, width: int, height: int
    ) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate Euler angles (pitch, yaw, roll) via solvePnP.

        Returns (pose_dict, rvec, tvec).
        rvec/tvec are exposed for T3-1 (solvePnP full projection correction).
        Both are None when solvePnP fails.
        """
        model_points = np.array([
            (0.0,    0.0,    0.0),       # Nose tip
            (0.0,   -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),     # Left eye outer corner
            (225.0,  170.0, -135.0),     # Right eye outer corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0,  -150.0, -125.0),    # Right mouth corner
        ])
        image_points = np.array([
            (landmarks[1].x   * width, landmarks[1].y   * height),
            (landmarks[199].x * width, landmarks[199].y * height),
            (landmarks[33].x  * width, landmarks[33].y  * height),
            (landmarks[263].x * width, landmarks[263].y * height),
            (landmarks[61].x  * width, landmarks[61].y  * height),
            (landmarks[291].x * width, landmarks[291].y * height),
        ], dtype="double")

        focal   = float(width)
        cam_mat = np.array([[focal, 0, width / 2],
                            [0, focal, height / 2],
                            [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        try:
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, cam_mat, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, None, None
        except cv2.error:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, None, None

        rmat, _ = cv2.Rodrigues(rvec)
        proj    = np.hstack((rmat, tvec))
        decomp  = cv2.decomposeProjectionMatrix(proj)
        euler   = decomp[6] if len(decomp) >= 7 else [[0], [0], [0]]

        pose = {
            "pitch": float(euler[0][0]),
            "yaw":   float(euler[1][0]),
            "roll":  float(euler[2][0]),
        }
        return pose, rvec, tvec

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _landmark_dist_px(self, landmarks, idx_a: int, idx_b: int, width: int, height: int) -> float:
        """Euclidean pixel distance between two MediaPipe landmarks."""
        a, b = landmarks[idx_a], landmarks[idx_b]
        return float(np.sqrt(((a.x - b.x) * width) ** 2 + ((a.y - b.y) * height) ** 2))

    def _landmark_horiz_px(self, landmarks, idx_a: int, idx_b: int, width: int) -> float:
        """Horizontal-only pixel distance between two MediaPipe landmarks.

        IMP D: ICD, OCD, and fissure are anatomically horizontal measurements.
        Using Euclidean distance inflates them when there is any vertical
        landmark offset (roll, asymmetry, detection noise).  Because the
        mm/px scale is derived from these distances, Euclidean inflation
        deflates the scale and therefore overestimates PD.  Using abs(dx)
        removes this bias without any loss of information — the horizontal
        component is what we actually want to compare against the known
        anatomical reference in mm.
        """
        a, b = landmarks[idx_a], landmarks[idx_b]
        return abs(float(a.x - b.x)) * width

    def _eye_openness(self, landmarks, upper_idx: int, lower_idx: int, height: int) -> float:
        """Vertical distance between upper/lower lid landmarks (normalised by height)."""
        return abs(landmarks[upper_idx].y - landmarks[lower_idx].y)

    def _reject(self, landmarks, width, height, pose, quality, reason: str) -> IrisData:
        """Build a minimal IrisData marked as unusable."""
        logger.debug("Frame rejected: %s", reason)
        li  = landmarks[self.LEFT_IRIS_CENTER]
        ri  = landmarks[self.RIGHT_IRIS_CENTER]
        return IrisData(
            left_iris_center=(li.x, li.y, li.z),
            right_iris_center=(ri.x, ri.y, ri.z),
            left_iris_pixel=(int(li.x * width), int(li.y * height)),
            right_iris_pixel=(int(ri.x * width), int(ri.y * height)),
            left_iris_diameter_px=0.0,
            right_iris_diameter_px=0.0,
            face_width_px=0.0,
            nose_bridge_x=0.0,
            confidence=0.0,
            head_pose=pose,
            quality_score=quality["blur"],
            iris_canthus_ratio=0.0,
            pose_symmetry={"horizontal": 0.5, "vertical": 0.5},
            is_usable=False,
            reject_reason=reason,
            landmarks_raw=landmarks,
            face_depth_z=0.0,
            left_eye_openness=0.0,
            right_eye_openness=0.0,
        )

    def _calculate_robust_diameter(self, landmarks, indices: list, width: int, height: int) -> float:
        """Robust iris diameter: median of edge-to-centre distances × 2."""
        centre = landmarks[indices[0]]
        cx = centre.x * width
        cy = centre.y * height
        radii = [
            float(np.linalg.norm(
                np.array([landmarks[i].x * width - cx,
                          landmarks[i].y * height - cy])
            ))
            for i in indices[1:]
        ]
        return 2.0 * float(np.median(radii)) if radii else 0.0

    def _ellipse_iris_centre(
        self, landmarks, indices: list, width: int, height: int
    ) -> Tuple[float, float, float, float]:
        """Sub-pixel iris centre and diameter via ellipse fitting (T3-2).

        Fits an ellipse to the 5 iris landmarks (centre + 4 boundary points).
        The ellipse centre is geometrically more stable than the single MediaPipe
        centre landmark (~0.5–1px jitter per frame).

        Returns (cx_px, cy_px, diameter_px, ok) where ok=1.0 on success, 0.0 on fallback.
        Falls back to the MediaPipe centre point if fitting fails (e.g. degenerate
        points when glasses occlude part of the iris boundary).
        """
        pts = np.array(
            [[landmarks[i].x * width, landmarks[i].y * height] for i in indices],
            dtype=np.float32,
        )
        centre_mp = pts[0]  # MediaPipe centre as fallback

        # fitEllipse requires ≥ 5 points — we have exactly 5
        try:
            (ex, ey), (ma, mb), angle = cv2.fitEllipse(pts)
            # Sanity check: ellipse centre should be within ~5px of MediaPipe centre
            dist_from_mp = float(np.linalg.norm(np.array([ex, ey]) - centre_mp))
            major_axis   = max(ma, mb)
            minor_axis   = min(ma, mb)
            # T3-H: face-relative bounds instead of fixed pixel caps.
            # The iris is physiologically 2%–12% of face width at any resolution.
            # Fixed pixel caps (e.g. 80px) break at 1280px+ (iris can reach ~55px).
            iris_min_px = width * 0.02   # ~2% of frame width — smallest plausible iris
            iris_max_px = width * 0.12   # ~12% of frame width — largest plausible iris
            if (dist_from_mp > 5.0
                    or major_axis < iris_min_px
                    or major_axis > iris_max_px):
                raise ValueError(
                    f"Degenerate ellipse: dist={dist_from_mp:.1f} "
                    f"axes=({ma:.1f},{mb:.1f}) "
                    f"bounds=[{iris_min_px:.1f},{iris_max_px:.1f}]"
                )
            # BUG B fix: use geometric mean sqrt(major * minor) as the iris diameter.
            # The iris is a circle; under any head roll or slight tilt it projects as
            # an ellipse. The major axis alone overestimates the true diameter, which
            # deflates the mm/px scale and therefore deflates PD. The geometric mean
            # is the correct estimator for the diameter of a circle projected as an
            # ellipse (it equals the true diameter when the circle is frontal, and
            # falls between major and minor otherwise — always closer to truth).
            geom_diameter = float(np.sqrt(major_axis * minor_axis))
            return float(ex), float(ey), geom_diameter, 1.0
        except (cv2.error, ValueError) as e:
            logger.debug("Ellipse fit fallback: %s", e)
            # Fallback: MediaPipe centre + robust diameter
            diam = self._calculate_robust_diameter(landmarks, indices, width, height)
            return float(centre_mp[0]), float(centre_mp[1]), diam, 0.0

    def estimate_age_group(self, landmarks, width: int, height: int) -> str:
        """Estimate age group from facial proportions (T3-3B).

        Children have measurably different facial geometry than adults:
          1. Eye-height ratio  — eyes are larger relative to face height, sit lower
          2. Forehead ratio    — taller forehead relative to face height
          3. Face roundness    — face width / face height closer to 1.0 in children
          4. Nose length ratio — shorter nose relative to face height

        Each ratio is scored 0–1 (1 = most child-like), weighted, and summed
        into a child-likelihood score that maps to one of four age groups.
        Returns "adult" on any estimation failure (T1-2 — "unknown" is not a
        valid IRIS_DIAMETER_BY_AGE key).

        Landmark references (MediaPipe 478-point canonical face model):
          10  = top of forehead
          152 = chin tip
          1   = nose tip
          168 = nose bridge (between eyes)
          33  = right eye outer corner
          263 = left eye outer corner
          61  = right mouth corner
          291 = left mouth corner
          159 = left upper lid (eye height proxy)
          145 = left lower lid
        """
        try:
            # --- Raw landmark positions ---
            forehead_y   = landmarks[10].y  * height    # top of forehead
            chin_y       = landmarks[152].y * height    # bottom of chin
            nose_tip_y   = landmarks[1].y   * height    # nose tip
            nose_bridge_y= landmarks[168].y * height    # nose bridge
            r_eye_x      = landmarks[33].x  * width
            l_eye_x      = landmarks[263].x * width
            r_mouth_x    = landmarks[61].x  * width
            l_mouth_x    = landmarks[291].x * width
            eye_upper_y  = landmarks[159].y * height
            eye_lower_y  = landmarks[145].y * height

            face_height  = abs(chin_y - forehead_y)
            face_width   = abs(l_eye_x - r_eye_x)   # inter-ocular as face width proxy
            mouth_width  = abs(l_mouth_x - r_mouth_x)

            if face_height < 10 or face_width < 5:
                return "adult"   # T1-2: face too small to score — default to adult

            # --- Feature 1: eye-height ratio ---
            # Children's eyes are ~12–14% of face height; adults ~9–11%
            eye_height_px    = abs(eye_upper_y - eye_lower_y)
            eye_height_ratio = eye_height_px / face_height
            # Score: 0 at 0.09 (adult), 1 at 0.14 (young child)
            eye_score = float(np.clip((eye_height_ratio - 0.09) / (0.14 - 0.09), 0.0, 1.0))

            # --- Feature 2: forehead ratio ---
            # Children have taller foreheads (nose bridge sits lower on face)
            # Ratio = (nose_bridge_y - forehead_y) / face_height
            # Adults ~0.38–0.42, children ~0.45–0.52
            forehead_ratio = (nose_bridge_y - forehead_y) / face_height
            forehead_score = float(np.clip((forehead_ratio - 0.38) / (0.52 - 0.38), 0.0, 1.0))

            # --- Feature 3: face roundness ---
            # Children: face_width / face_height closer to 0.65–0.75
            # Adults: 0.50–0.60
            roundness = face_width / face_height
            roundness_score = float(np.clip((roundness - 0.50) / (0.75 - 0.50), 0.0, 1.0))

            # --- Feature 4: nose length ratio ---
            # Children have shorter noses: (nose_tip_y - nose_bridge_y) / face_height
            # Adults ~0.22–0.27, children ~0.14–0.20
            nose_ratio = (nose_tip_y - nose_bridge_y) / face_height
            # Invert: short nose → high child score
            nose_score = float(np.clip((0.27 - nose_ratio) / (0.27 - 0.14), 0.0, 1.0))

            # --- Feature 5: mouth-to-face-width ratio ---
            # Children's mouths are narrower relative to face width
            # Adults: mouth/face_width ~0.70–0.80, children: ~0.55–0.68
            mouth_ratio = mouth_width / face_width if face_width > 0 else 0.7
            mouth_score = float(np.clip((0.75 - mouth_ratio) / (0.75 - 0.55), 0.0, 1.0))

            # --- Weighted child-likelihood score ---
            child_score = (
                eye_score       * 0.30 +
                forehead_score  * 0.25 +
                roundness_score * 0.20 +
                nose_score      * 0.15 +
                mouth_score     * 0.10
            )

            logger.debug(
                "Age estimation: eye=%.2f forehead=%.2f round=%.2f nose=%.2f mouth=%.2f → score=%.2f",
                eye_score, forehead_score, roundness_score, nose_score, mouth_score, child_score,
            )

            if child_score >= 0.72:
                return "young_child"
            elif child_score >= 0.52:
                return "child"
            elif child_score >= 0.38:
                return "teen"
            else:
                return "adult"

        except Exception as e:
            logger.debug("Age estimation failed: %s", e)
            return "adult"   # T1-2: "unknown" is not a valid key — default to adult

    def _calculate_pose_symmetry(self, landmarks) -> Dict[str, float]:
        left_eye_x  = landmarks[33].x
        right_eye_x = landmarks[263].x
        nose_x      = landmarks[1].x
        denom_h = right_eye_x - left_eye_x
        horizontal  = (nose_x - left_eye_x) / denom_h if denom_h != 0 else 0.5

        nose_y      = landmarks[1].y
        eye_avg_y   = (landmarks[33].y + landmarks[263].y) / 2
        mouth_avg_y = (landmarks[61].y + landmarks[291].y) / 2
        denom_v = mouth_avg_y - eye_avg_y
        vertical    = (nose_y - eye_avg_y) / denom_v if denom_v != 0 else 0.5

        return {
            "horizontal": round(horizontal, 3),
            "vertical":   round(vertical,   3),
        }

    def _calculate_advanced_confidence(
        self,
        pose: dict,
        quality: dict,
        d1: float,
        d2: float,
        ellipse_ok: bool = False,
        pnp_ok: bool = False,
        camera_mode: bool = False,
    ) -> float:
        """Compute a 0–1 confidence score for this frame.

        T2-1 improvements over the old formula:
          - blur_score is continuous (0→1) instead of the old binary 1.0/0.5
          - ellipse_ok rewards sub-pixel iris centre fit (+0.05 bonus)
          - pnp_ok rewards successful 6-DOF pose correction (+0.05 bonus)
          - weights rebalanced to sum to 1.0 across the four base components

        Components (sum = 1.0 base):
          tilt_penalty  0.35  — yaw+pitch tilt penalty (was 0.40)
          blur_cont     0.30  — continuous Laplacian score (was binary 0.30)
          symmetry      0.25  — iris diameter symmetry (was 0.30)
          quality_gate  0.10  — binary brightness-in-range gate (was merged with blur)

        Bonuses (additive, capped at 1.0 total):
          +0.05 when ellipse fit succeeded (sub-pixel iris centre)
          +0.05 when solvePnP succeeded (full 6-DOF correction available)
        """
        adj_pitch = abs(pose["pitch"])
        if adj_pitch > 90:
            adj_pitch = abs(adj_pitch - 180)

        # 1. Tilt penalty — yaw + pitch (roll already gated to ≤20°)
        tilt_penalty = max(0.0, 1.0 - (abs(pose["yaw"]) + adj_pitch) / 40.0)

        # 2. Continuous blur score: map Laplacian variance to 0–1.
        # Use the appropriate MIN_BLUR floor depending on source (photo vs camera).
        # We treat 4× floor as "perfect sharpness" → score 1.0.
        blur_floor = self.MIN_BLUR_CAMERA if camera_mode else self.MIN_BLUR
        blur_raw = float(quality["blur"])
        blur_cont = float(np.clip((blur_raw - blur_floor) / (3.0 * blur_floor), 0.0, 1.0))

        # 3. Iris diameter symmetry (ratio of smaller/larger diameter)
        symmetry = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0

        # 4. Brightness quality gate (binary — either in range or not)
        quality_gate = 1.0 if quality["is_reliable"] else 0.0

        base = (
            tilt_penalty * 0.35
            + blur_cont  * 0.30
            + symmetry   * 0.25
            + quality_gate * 0.10
        )

        # Additive bonuses for better pose/iris data quality
        bonus = (0.05 if ellipse_ok else 0.0) + (0.05 if pnp_ok else 0.0)

        return round(min(1.0, base + bonus), 3)

    def close(self):
        self.face_mesh.close()
