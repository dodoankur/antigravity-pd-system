"""
Advanced PD Calculator Service
Implements Head Pose Correction and Cross-Validation Scaling
"""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from services.face_detection import IrisData
from services.reference_detection import ReferenceDetectionResult
from models.schemas import MeasurementMethod, ErrorMargin


@dataclass
class PDResult:
    overall_pd_mm: float
    left_pd_mm: float
    right_pd_mm: float
    method: MeasurementMethod
    model_used: str
    confidence_score: float
    error_margin: ErrorMargin
    disclaimer: str
    reference_detected: Optional[bool] = None
    reference_scale_factor: Optional[float] = None


class PDCalculatorService:
    # Industry constants
    AVERAGE_IRIS_DIAMETER_MM = 11.7
    
    def calculate_pd_with_iris_estimation(self, iris_data: IrisData) -> PDResult:
        # 1. Perspective Correction
        # Adjust diameter based on head yaw (profile view reduces perceived diameter)
        yaw_rad = np.deg2rad(iris_data.head_pose['yaw'])
        
        # Compensate for foreshortening
        corrected_left_dia = iris_data.left_iris_diameter_px / np.cos(yaw_rad)
        corrected_right_dia = iris_data.right_iris_diameter_px / np.cos(yaw_rad)
        
        avg_iris_px = (corrected_left_dia + corrected_right_dia) / 2
        
        # HYBRID SCALING: (Iris + Canthus stabilization)
        # Iris estimation (Primary anchor: 11.7mm)
        scale_iris = self.AVERAGE_IRIS_DIAMETER_MM / avg_iris_px
        
        # Canthus estimation (Secondary anchor: ~30mm across one eye corner to corner)
        # Helps stabilize scale at distances where iris pixels are few
        scale_canthus = 30.0 / iris_data.face_width_px if iris_data.face_width_px > 0 else scale_iris
        
        # Weighted Blend: 85% Iris (high accuracy), 15% Canthus (high stability)
        mm_per_pixel = (scale_iris * 0.85) + (scale_canthus * 0.15)
        
        # 2. PD Measurement
        left_px = np.array(iris_data.left_iris_pixel)
        right_px = np.array(iris_data.right_iris_pixel)
        
        # Euclidean distance in pixels
        pd_pixels = np.linalg.norm(left_px - right_px)
        
        # Apply perspective compensation to PD distance
        # If head is turned, actual PD is scaled by 1/cos(yaw)
        corrected_pd_mm = (pd_pixels / np.cos(yaw_rad)) * mm_per_pixel
        
        # 3. Monocular PD (Split)
        # Using 50/50 split for now
        left_pd_mm = corrected_pd_mm / 2
        right_pd_mm = corrected_pd_mm / 2
        
        # 4. Accuracy & Confidence
        conf = iris_data.confidence
        error_val = 1.0 if conf > 0.9 else 1.5 if conf > 0.7 else 2.5
        
        error_margin = ErrorMargin(
            value_mm=error_val,
            percentage=round((error_val/corrected_pd_mm)*100, 1),
            confidence_score=conf
        )

        return PDResult(
            overall_pd_mm=round(corrected_pd_mm, 1),
            left_pd_mm=round(left_pd_mm, 1),
            right_pd_mm=round(right_pd_mm, 1),
            method=MeasurementMethod.IRIS_ESTIMATION,
            model_used="MediaPipe v0.10 + Perspective Correction Engine",
            confidence_score=conf,
            error_margin=error_margin,
            disclaimer=self._generate_advanced_disclaimer(conf, iris_data.head_pose)
        )

    def calculate_pd_with_reference(self, iris_data: IrisData, ref: ReferenceDetectionResult) -> PDResult:
        mm_per_pixel = ref.scale_factor
        pd_px = np.linalg.norm(np.array(iris_data.left_iris_pixel) - np.array(iris_data.right_iris_pixel))
        
        # Even with reference, correct for head tilt
        yaw_rad = np.deg2rad(iris_data.head_pose['yaw'])
        corrected_pd_mm = (pd_px / np.cos(yaw_rad)) * mm_per_pixel
        
        return PDResult(
            overall_pd_mm=round(corrected_pd_mm, 1),
            left_pd_mm=round(corrected_pd_mm/2, 1),
            right_pd_mm=round(corrected_pd_mm/2, 1),
            method=MeasurementMethod.REFERENCE_OBJECT,
            model_used="Ref-Object + Tilt Compensation",
            confidence_score=iris_data.confidence,
            error_margin=ErrorMargin(value_mm=1.0, percentage=2.0, confidence_score=iris_data.confidence),
            disclaimer="Reference Object Mode: High Precision",
            reference_detected=True,
            reference_scale_factor=mm_per_pixel
        )

    def _generate_advanced_disclaimer(self, confidence, pose) -> str:
        warn = ""
        if abs(pose['yaw']) > 15: warn += "⚠️ Head turned too much. "
        if abs(pose['pitch']) > 15: warn += "⚠️ Looking too far up/down. "
        
        return f"{warn}Calculated using 3D Perspective Correction. Confidence: {confidence*100}%"
