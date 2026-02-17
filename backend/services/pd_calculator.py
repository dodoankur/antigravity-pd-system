"""
PD Calculator Service
Calculates pupil distance with and without reference objects
"""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from services.face_detection import IrisData, FaceDetectionService
from services.reference_detection import ReferenceDetectionResult
from models.schemas import MeasurementMethod, ErrorMargin


@dataclass
class PDResult:
    """Complete PD measurement result"""
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
    """
    Service for calculating Pupil Distance (PD) measurements
    
    Supports two methods:
    1. Iris Estimation: Uses average human iris diameter (11.7mm) to calculate scale
    2. Reference Object: Uses detected reference object for more accurate scale
    
    Calculations:
    - Overall PD: Distance between left and right iris centers
    - Left/Right PD (Monocular): Distance from each iris to the center of nose bridge
    """
    
    # Average human iris diameter in mm (well-documented medical constant)
    AVERAGE_IRIS_DIAMETER_MM = 11.7
    IRIS_DIAMETER_STD_MM = 0.5  # Standard deviation
    
    # Error margin configurations
    IRIS_ESTIMATION_ERROR_MM = 1.5
    IRIS_ESTIMATION_ERROR_PERCENT = 3.0
    IRIS_ESTIMATION_BASE_CONFIDENCE = 0.88  # Based on research (91% accuracy claim)
    
    REFERENCE_OBJECT_ERROR_MM = 1.0
    REFERENCE_OBJECT_ERROR_PERCENT = 2.0
    REFERENCE_OBJECT_BASE_CONFIDENCE = 0.92
    
    def calculate_pd_with_iris_estimation(
        self, 
        iris_data: IrisData
    ) -> PDResult:
        """
        Calculate PD using iris diameter estimation method
        
        This method uses the average human iris diameter (11.7mm) to calculate
        a pixels-per-mm scale factor, then measures the interpupillary distance.
        
        Args:
            iris_data: Detected iris data from FaceDetectionService
            
        Returns:
            PDResult with measurements and error margins
        """
        # Calculate scale factor from iris diameter
        avg_iris_diameter_px = (iris_data.left_iris_diameter_px + iris_data.right_iris_diameter_px) / 2
        mm_per_pixel = self.AVERAGE_IRIS_DIAMETER_MM / avg_iris_diameter_px
        
        # Calculate overall PD (distance between iris centers)
        left_px = np.array(iris_data.left_iris_pixel)
        right_px = np.array(iris_data.right_iris_pixel)
        pd_pixels = np.linalg.norm(left_px - right_px)
        overall_pd_mm = pd_pixels * mm_per_pixel
        
        # Calculate monocular PD (from each eye to estimated center)
        # Center is approximated as midpoint between irises (nose bridge)
        center_px = (left_px + right_px) / 2
        left_pd_mm = np.linalg.norm(left_px - center_px) * mm_per_pixel
        right_pd_mm = np.linalg.norm(right_px - center_px) * mm_per_pixel
        
        # Adjust confidence based on iris detection quality
        detection_confidence = iris_data.confidence
        overall_confidence = self.IRIS_ESTIMATION_BASE_CONFIDENCE * detection_confidence
        
        # Calculate error margin
        # Higher confidence reduces error margin slightly
        error_multiplier = 1.0 + (1.0 - detection_confidence) * 0.5
        error_mm = self.IRIS_ESTIMATION_ERROR_MM * error_multiplier
        error_percent = self.IRIS_ESTIMATION_ERROR_PERCENT * error_multiplier
        
        error_margin = ErrorMargin(
            value_mm=round(error_mm, 2),
            percentage=round(error_percent, 1),
            confidence_score=round(overall_confidence, 3)
        )
        
        disclaimer = self._generate_disclaimer(
            MeasurementMethod.IRIS_ESTIMATION,
            error_margin,
            overall_confidence
        )
        
        return PDResult(
            overall_pd_mm=round(overall_pd_mm, 1),
            left_pd_mm=round(left_pd_mm, 1),
            right_pd_mm=round(right_pd_mm, 1),
            method=MeasurementMethod.IRIS_ESTIMATION,
            model_used="MediaPipe Face Mesh v0.10.9 with Iris Refinement",
            confidence_score=round(overall_confidence, 3),
            error_margin=error_margin,
            disclaimer=disclaimer,
            reference_detected=None,
            reference_scale_factor=None
        )
    
    def calculate_pd_with_reference(
        self, 
        iris_data: IrisData,
        reference_result: ReferenceDetectionResult
    ) -> PDResult:
        """
        Calculate PD using reference object for scale calibration
        
        This method uses a detected reference object (credit card, coin, ruler)
        to determine the pixels-per-mm scale, providing more accurate results.
        
        Args:
            iris_data: Detected iris data from FaceDetectionService
            reference_result: Detected reference object with scale factor
            
        Returns:
            PDResult with measurements and error margins
        """
        mm_per_pixel = reference_result.scale_factor
        
        # Calculate overall PD
        left_px = np.array(iris_data.left_iris_pixel)
        right_px = np.array(iris_data.right_iris_pixel)
        pd_pixels = np.linalg.norm(left_px - right_px)
        overall_pd_mm = pd_pixels * mm_per_pixel
        
        # Calculate monocular PD
        center_px = (left_px + right_px) / 2
        left_pd_mm = np.linalg.norm(left_px - center_px) * mm_per_pixel
        right_pd_mm = np.linalg.norm(right_px - center_px) * mm_per_pixel
        
        # Combine confidence from face detection and reference detection
        detection_confidence = iris_data.confidence
        reference_confidence = reference_result.confidence
        overall_confidence = (
            self.REFERENCE_OBJECT_BASE_CONFIDENCE * 
            detection_confidence * 
            reference_confidence
        )
        
        # Calculate error margin (lower for reference-based)
        error_multiplier = 1.0 + (1.0 - min(detection_confidence, reference_confidence)) * 0.3
        error_mm = self.REFERENCE_OBJECT_ERROR_MM * error_multiplier
        error_percent = self.REFERENCE_OBJECT_ERROR_PERCENT * error_multiplier
        
        error_margin = ErrorMargin(
            value_mm=round(error_mm, 2),
            percentage=round(error_percent, 1),
            confidence_score=round(overall_confidence, 3)
        )
        
        disclaimer = self._generate_disclaimer(
            MeasurementMethod.REFERENCE_OBJECT,
            error_margin,
            overall_confidence,
            reference_result.object_type.value
        )
        
        return PDResult(
            overall_pd_mm=round(overall_pd_mm, 1),
            left_pd_mm=round(left_pd_mm, 1),
            right_pd_mm=round(right_pd_mm, 1),
            method=MeasurementMethod.REFERENCE_OBJECT,
            model_used="MediaPipe Face Mesh v0.10.9 + Reference Object Calibration",
            confidence_score=round(overall_confidence, 3),
            error_margin=error_margin,
            disclaimer=disclaimer,
            reference_detected=True,
            reference_scale_factor=round(mm_per_pixel, 6)
        )
    
    def _generate_disclaimer(
        self,
        method: MeasurementMethod,
        error_margin: ErrorMargin,
        confidence: float,
        reference_type: Optional[str] = None
    ) -> str:
        """
        Generate detailed disclaimer with method, accuracy, and confidence info
        """
        if method == MeasurementMethod.IRIS_ESTIMATION:
            method_desc = "Iris Diameter Estimation (using average human iris size of 11.7mm)"
        else:
            ref_name = reference_type.replace("_", " ").title() if reference_type else "Reference Object"
            method_desc = f"Reference Object Calibration ({ref_name})"
        
        confidence_level = "High" if confidence >= 0.85 else "Moderate" if confidence >= 0.70 else "Low"
        
        disclaimer = (
            f"Measurement Method: {method_desc}\n"
            f"Estimated Accuracy: ±{error_margin.value_mm}mm (±{error_margin.percentage}%)\n"
            f"Confidence Score: {error_margin.confidence_score:.1%} ({confidence_level})\n"
            f"\n"
            f"⚠️ IMPORTANT: This is an estimated measurement intended for general guidance. "
            f"For prescription eyewear, please verify your PD with a qualified optician "
            f"or eye care professional. Factors such as camera angle, lighting, and "
            f"image quality may affect accuracy."
        )
        
        return disclaimer
