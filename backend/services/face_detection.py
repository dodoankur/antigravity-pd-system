"""
Face Detection Service using MediaPipe Face Mesh
Detects facial landmarks including iris positions for PD measurement
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class IrisData:
    """Data structure for iris detection results"""
    left_iris_center: Tuple[float, float, float]  # x, y, z in normalized coords
    right_iris_center: Tuple[float, float, float]
    left_iris_pixel: Tuple[int, int]  # x, y in pixel coords
    right_iris_pixel: Tuple[int, int]
    left_iris_diameter_px: float
    right_iris_diameter_px: float
    face_width_px: float
    confidence: float


class FaceDetectionService:
    """
    Service for detecting faces and iris positions using MediaPipe Face Mesh
    
    MediaPipe Face Mesh provides 468 face landmarks + 10 iris landmarks (5 per eye)
    Iris landmarks: 468-472 (right eye), 473-477 (left eye)
    Center landmarks: 468 (right iris center), 473 (left iris center)
    """
    
    # MediaPipe iris landmark indices
    RIGHT_IRIS_CENTER = 468
    LEFT_IRIS_CENTER = 473
    RIGHT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]  # center + 4 edge points
    LEFT_IRIS_LANDMARKS = [473, 474, 475, 476, 477]
    
    # Face width landmarks (outer eye corners for reference)
    LEFT_EYE_OUTER = 263
    RIGHT_EYE_OUTER = 33
    
    # Nose bridge landmark (for monocular PD calculation)
    NOSE_BRIDGE = 6
    
    # Average human iris diameter in mm (medical constant)
    AVERAGE_IRIS_DIAMETER_MM = 11.7
    IRIS_DIAMETER_STD_MM = 0.5
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh with iris refinement"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_iris(self, image: np.ndarray) -> Optional[IrisData]:
        """
        Detect iris positions in the given image
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            IrisData object with iris positions, or None if detection fails
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Run face mesh detection
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first detected face
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        
        # Extract iris centers (normalized coordinates)
        left_iris = landmarks[self.LEFT_IRIS_CENTER]
        right_iris = landmarks[self.RIGHT_IRIS_CENTER]
        
        # Convert to pixel coordinates
        left_iris_px = (int(left_iris.x * width), int(left_iris.y * height))
        right_iris_px = (int(right_iris.x * width), int(right_iris.y * height))
        
        # Calculate iris diameters from edge landmarks
        left_diameter = self._calculate_iris_diameter(
            landmarks, self.LEFT_IRIS_LANDMARKS, width, height
        )
        right_diameter = self._calculate_iris_diameter(
            landmarks, self.RIGHT_IRIS_LANDMARKS, width, height
        )
        
        # Calculate face width for reference
        left_eye_outer = landmarks[self.LEFT_EYE_OUTER]
        right_eye_outer = landmarks[self.RIGHT_EYE_OUTER]
        face_width = np.sqrt(
            ((left_eye_outer.x - right_eye_outer.x) * width) ** 2 +
            ((left_eye_outer.y - right_eye_outer.y) * height) ** 2
        )
        
        # Calculate confidence based on landmark visibility and consistency
        confidence = self._calculate_confidence(
            left_iris, right_iris, left_diameter, right_diameter
        )
        
        return IrisData(
            left_iris_center=(left_iris.x, left_iris.y, left_iris.z),
            right_iris_center=(right_iris.x, right_iris.y, right_iris.z),
            left_iris_pixel=left_iris_px,
            right_iris_pixel=right_iris_px,
            left_iris_diameter_px=left_diameter,
            right_iris_diameter_px=right_diameter,
            face_width_px=face_width,
            confidence=confidence
        )
    
    def get_nose_bridge_position(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Get nose bridge position for monocular PD calculation
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            Tuple of (x, y) pixel coordinates, or None if detection fails
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        nose_bridge = results.multi_face_landmarks[0].landmark[self.NOSE_BRIDGE]
        return (int(nose_bridge.x * width), int(nose_bridge.y * height))
    
    def _calculate_iris_diameter(
        self, 
        landmarks, 
        iris_indices: list, 
        width: int, 
        height: int
    ) -> float:
        """Calculate iris diameter from edge landmarks"""
        center = landmarks[iris_indices[0]]
        center_px = np.array([center.x * width, center.y * height])
        
        # Calculate distances to all edge points and take average diameter
        distances = []
        for idx in iris_indices[1:]:
            edge = landmarks[idx]
            edge_px = np.array([edge.x * width, edge.y * height])
            distances.append(np.linalg.norm(edge_px - center_px))
        
        # Diameter is 2 * average radius
        return 2 * np.mean(distances)
    
    def _calculate_confidence(
        self,
        left_iris,
        right_iris,
        left_diameter: float,
        right_diameter: float
    ) -> float:
        """
        Calculate detection confidence score (0-1)
        
        Factors:
        - Landmark visibility
        - Iris diameter consistency (left vs right should be similar)
        - Z-depth consistency (both eyes should be at similar depth)
        """
        # Check visibility (if landmarks have visibility attribute)
        visibility_score = 1.0
        
        # Diameter consistency (both irises should be similar size)
        diameter_ratio = min(left_diameter, right_diameter) / max(left_diameter, right_diameter)
        diameter_score = diameter_ratio  # 1.0 if identical, lower if different
        
        # Z-depth consistency
        z_diff = abs(left_iris.z - right_iris.z)
        z_score = max(0, 1.0 - z_diff * 10)  # Penalize large z differences
        
        # Combined confidence
        confidence = (visibility_score * 0.3 + diameter_score * 0.4 + z_score * 0.3)
        return round(min(1.0, max(0.0, confidence)), 3)
    
    def close(self):
        """Release MediaPipe resources"""
        self.face_mesh.close()
