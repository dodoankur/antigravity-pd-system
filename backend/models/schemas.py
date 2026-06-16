"""
Pydantic schemas for PD Measurement API
"""
from pydantic import BaseModel
from typing import Optional, Literal
from enum import Enum


class ReferenceType(str, Enum):
    NONE = "none"
    CREDIT_CARD = "credit_card"
    COIN_GBP_1P = "coin_gbp_1p"
    COIN_GBP_2P = "coin_gbp_2p"
    COIN_GBP_5P = "coin_gbp_5p"
    COIN_GBP_10P = "coin_gbp_10p"
    COIN_GBP_20P = "coin_gbp_20p"
    COIN_GBP_50P = "coin_gbp_50p"
    COIN_GBP_1 = "coin_gbp_1"
    COIN_GBP_2 = "coin_gbp_2"
    RULER = "ruler"


class MeasurementMethod(str, Enum):
    IRIS_ESTIMATION = "iris_estimation"
    REFERENCE_OBJECT = "reference_object"


class PDMeasurementRequest(BaseModel):
    """Request model for PD measurement"""
    reference_type: ReferenceType = ReferenceType.NONE


class ErrorMargin(BaseModel):
    """Error margin details for a measurement"""
    value_mm: float
    percentage: float
    confidence_score: float


class PDMeasurementResult(BaseModel):
    """Result model for PD measurement"""
    model_config = {"protected_namespaces": ()}
    
    overall_pd_mm: float
    left_pd_mm: float
    right_pd_mm: float
    
    # Method and confidence info
    method: MeasurementMethod
    model_used: str
    confidence_score: float
    
    # Error margins
    error_margin: ErrorMargin
    
    # Detailed disclaimer
    disclaimer: str
    
    # Debug info (optional)
    face_detected: bool = True
    eyes_detected: bool = True
    reference_detected: Optional[bool] = None
    reference_scale_factor: Optional[float] = None
    # NEW: True when |left_pd - right_pd| > 4mm — suggests retake or optician visit
    asymmetry_warning: bool = False
    # NEW: how many frames were accepted / rejected in a batch
    frames_accepted: Optional[int] = None
    frames_rejected: Optional[int] = None
    # NEW: age group used for iris diameter calibration (auto-detected or user-selected)
    age_group_used: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
