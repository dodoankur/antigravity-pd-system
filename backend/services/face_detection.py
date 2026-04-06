"""
Advanced Face Detection Service using MediaPipe Face Mesh
Includes Perspective Correction, Quality Checks, and Robust Iris Fitting
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass


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
    head_pose: Dict[str, float]  # pitch, yaw, roll
    quality_score: float         # brightness, blurriness
    iris_canthus_ratio: float    # for self-calibration
    pose_symmetry: Dict[str, float] # horizontal/vertical symmetry ratios


class FaceDetectionService:
    """
    Advanced Service for high-precision PD measurement features:
    1. MediaPipe Iris Refinement
    2. Head Pose / Perspective Correction
    3. Image Quality Assessment (Blur/Lighting)
    4. Cross-validation via Canthus landmarks
    """
    
    # Landmark Indices
    RIGHT_IRIS_CENTER = 468
    LEFT_IRIS_CENTER = 473
    RIGHT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]
    LEFT_IRIS_LANDMARKS = [473, 474, 475, 476, 477]
    
    # Canthus (Eye Corners) for cross-validation
    LEFT_EYE_INNER = 463
    LEFT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 33
    RIGHT_EYE_OUTER = 133
    
    # Pose Estimation Landmarks
    POSE_LANDMARKS = [33, 263, 1, 61, 291, 199] # Left Eye, Right Eye, Nose tip, Mouth corners, Chin
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7, # Increased for precision
            min_tracking_confidence=0.7
        )

    def assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Calculates brightness and blurriness scores"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Lapalacian variance for blur detection (Higher = Sharper)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Average brightness (0-255)
        brightness = np.mean(gray)
        
        return {
            "blur": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "is_reliable": blur_score > 100 and 50 < brightness < 220
        }

    def estimate_head_pose(self, landmarks, width: int, height: int) -> Dict[str, float]:
        """Estimates Euler angles (Pitch, Yaw, Roll) for perspective correction"""
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # 2D image points from landmarks
        image_points = np.array([
            (landmarks[1].x * width, landmarks[1].y * height),    # Nose tip
            (landmarks[199].x * width, landmarks[199].y * height),# Chin
            (landmarks[33].x * width, landmarks[33].y * height),  # Left eye left
            (landmarks[263].x * width, landmarks[263].y * height),# Right eye right
            (landmarks[61].x * width, landmarks[61].y * height),  # Left mouth
            (landmarks[291].x * width, landmarks[291].y * height) # Right mouth
        ], dtype="double")

        camera_matrix = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        except cv2.error:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        
        # Convert to Euler angles
        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        decomp = cv2.decomposeProjectionMatrix(proj_matrix)
        
        if len(decomp) >= 7:
            euler_angles = decomp[6]
        else:
            euler_angles = decomp[1] if decomp[1].size == 3 else [0, 0, 0] # Avoid crashing on unexpected fallback size
            
        return {
            "pitch": float(euler_angles[0][0]),
            "yaw": float(euler_angles[1][0]),
            "roll": float(euler_angles[2][0])
        }

    def detect_iris(self, image: np.ndarray) -> Optional[IrisData]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Pre-check quality
        quality = self.assess_quality(image)
        
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Head Pose for perspective correction
        pose = self.estimate_head_pose(landmarks, width, height)
        
        nose_bridge = landmarks[168]
        nose_bridge_x = nose_bridge.x * width
        
        # Iris Detection
        left_iris = landmarks[self.LEFT_IRIS_CENTER]
        right_iris = landmarks[self.RIGHT_IRIS_CENTER]
        left_iris_px = (int(left_iris.x * width), int(left_iris.y * height))
        right_iris_px = (int(right_iris.x * width), int(right_iris.y * height))
        
        # Robust Iris Diameter (RANSAC alternative: using median of edge distances)
        left_diameter = self._calculate_robust_diameter(landmarks, self.LEFT_IRIS_LANDMARKS, width, height)
        right_diameter = self._calculate_robust_diameter(landmarks, self.RIGHT_IRIS_LANDMARKS, width, height)
        
        # Cross-Validation: Canthus (Eye corner) distance
        li = landmarks[self.LEFT_EYE_INNER]
        lo = landmarks[self.LEFT_EYE_OUTER]
        canthus_dist_px = np.sqrt(((li.x-lo.x)*width)**2 + ((li.y-lo.y)*height)**2)
        iris_canthus_ratio = (left_diameter / canthus_dist_px) if canthus_dist_px > 0 else 0

        # 2. Pose Symmetry (Alignment with Frontend logic)
        pose_symmetry = self._calculate_pose_symmetry(landmarks)

        # Confidence Scoring
        confidence = self._calculate_advanced_confidence(pose, quality, left_diameter, right_diameter)
        
        return IrisData(
            left_iris_center=(left_iris.x, left_iris.y, left_iris.z),
            right_iris_center=(right_iris.x, right_iris.y, right_iris.z),
            left_iris_pixel=left_iris_px,
            right_iris_pixel=right_iris_px,
            left_iris_diameter_px=left_diameter,
            right_iris_diameter_px=right_diameter,
            face_width_px=canthus_dist_px, # Used eye width for scaling anchor
            nose_bridge_x=nose_bridge_x,
            confidence=confidence,
            head_pose=pose,
            quality_score=quality['blur'],
            iris_canthus_ratio=iris_canthus_ratio,
            pose_symmetry=pose_symmetry
        )

    def _calculate_pose_symmetry(self, landmarks) -> Dict[str, float]:
        """Calculates horizontal and vertical symmetry ratios to match frontend logic"""
        # Horizontal Symmetry (Yaw)
        left_eye_x = landmarks[33].x
        right_eye_x = landmarks[263].x
        nose_x = landmarks[1].x
        
        horizontal_symmetry = (nose_x - left_eye_x) / (right_eye_x - left_eye_x) if (right_eye_x - left_eye_x) != 0 else 0.5
        
        # Vertical Symmetry (Pitch)
        nose_y = landmarks[1].y
        eye_avg_y = (landmarks[33].y + landmarks[263].y) / 2
        mouth_avg_y = (landmarks[61].y + landmarks[291].y) / 2
        
        vertical_symmetry = (nose_y - eye_avg_y) / (mouth_avg_y - eye_avg_y) if (mouth_avg_y - eye_avg_y) != 0 else 0.5
        
        return {
            "horizontal": round(horizontal_symmetry, 3),
            "vertical": round(vertical_symmetry, 3)
        }

    def _calculate_robust_diameter(self, landmarks, indices, width, height) -> float:
        center = landmarks[indices[0]]
        center_px = np.array([center.x * width, center.y * height])
        radii = []
        for i in indices[1:]:
            p = np.array([landmarks[i].x * width, landmarks[i].y * height])
            radii.append(np.linalg.norm(p - center_px))
        # Use median to handle outlier landmark jitter
        return 2 * np.median(radii)

    def _calculate_advanced_confidence(self, pose, quality, d1, d2) -> float:
        # Penalize for head tilt
        adjusted_pitch = pose['pitch']
        if adjusted_pitch > 90:
            adjusted_pitch -= 180
        elif adjusted_pitch < -90:
            adjusted_pitch += 180
            
        tilt_penalty = max(0, 1.0 - (abs(pose['yaw']) + abs(adjusted_pitch)) / 40.0)
        # Quality score
        quality_score = 1.0 if quality['is_reliable'] else 0.5
        # Symmetry check
        symmetry = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0
        
        return round((tilt_penalty * 0.4 + quality_score * 0.3 + symmetry * 0.3), 3)

    def close(self):
        self.face_mesh.close()
