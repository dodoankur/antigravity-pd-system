"""
PD Measurement API - FastAPI Backend
Provides endpoints for measuring pupil distance from face images
"""
import io
import cv2
import numpy as np
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from models.schemas import (
    ReferenceType, 
    PDMeasurementResult, 
    HealthResponse, 
    ErrorResponse,
    MeasurementMethod,
    ErrorMargin
)
from services.face_detection import FaceDetectionService
from services.reference_detection import ReferenceDetectionService
from services.pd_calculator import PDCalculatorService


# API Version
API_VERSION = "1.0.0"

# Initialize FastAPI app
app = FastAPI(
    title="PD Measurement API",
    description="API for measuring pupil distance from face images",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
face_detection_service = FaceDetectionService()
reference_detection_service = ReferenceDetectionService()
pd_calculator_service = PDCalculatorService()


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(status="healthy", version=API_VERSION)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version=API_VERSION)


@app.post("/api/pd/measure", response_model=PDMeasurementResult)
async def measure_pd(
    image: UploadFile = File(..., description="Face image with open eyes"),
    reference_type: str = Form(default="none", description="Type of reference object in image")
):
    """
    Measure pupil distance from an uploaded face image
    """
    # 1. Process the single image
    pd_result = await _process_single_image(image, reference_type)
    
    # 2. Return as response model
    return pd_result


@app.post("/api/pd/measure-batch", response_model=PDMeasurementResult)
async def measure_pd_batch(
    images: List[UploadFile] = File(..., description="List of face images for median calculation"),
    reference_type: str = Form(default="none", description="Type of reference object in image")
):
    """
    Measure pupil distance from multiple images and return the median result for higher stability.
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")

    results = []
    errors = []

    for img in images:
        try:
            res = await _process_single_image(img, reference_type)
            results.append(res)
        except HTTPException as e:
            errors.append(e.detail)
        except Exception as e:
            errors.append(str(e))

    if not results:
        raise HTTPException(
            status_code=422, 
            detail=f"Failed to process any images in the batch. Errors: {'; '.join(errors[:3])}"
        )

    # Calculate median of the overall_pd_mm
    pd_values = sorted([r.overall_pd_mm for r in results])
    median_pd = pd_values[len(pd_values) // 2]
    
    # Pick the representative result (the one closest to median or just the first successful one with updated median)
    final_result = results[0]
    final_result.overall_pd_mm = median_pd
    final_result.left_pd_mm = median_pd / 2
    final_result.right_pd_mm = median_pd / 2
    final_result.model_used = f"Batch Processing Median (from {len(results)} frames)"
    final_result.disclaimer = f"Calculated using {len(results)}-frame batch for maximum stability. {final_result.disclaimer}"

    return final_result


async def _process_single_image(image: UploadFile, reference_type: str) -> PDMeasurementResult:
    """Helper method to process a single image and return PD result"""
    # Validate image type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file (JPEG, PNG, WebP)."
        )
    
    try:
        # Read and convert image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary (handle RGBA, etc.)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Convert to OpenCV format (BGR)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )
    
    # Detect face and iris
    iris_data = face_detection_service.detect_iris(cv_image)
    
    if iris_data is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in image. Please upload a clear, front-facing photo with both eyes visible."
        )
    
    # ROBUST VALIDATION: Align with frontend symmetry and handle Euler flip noise
    # 1. Symmetry check (Primary - identical logic to frontend gating)
    sym = iris_data.pose_symmetry
    if not (0.35 <= sym['horizontal'] <= 0.65) or not (0.30 <= sym['vertical'] <= 0.70):
        print(f"DEBUG: Symmetry Check Failed - Horizontal: {sym['horizontal']}, Vertical: {sym['vertical']}")
        raise HTTPException(
            status_code=422,
            detail=f"Head tilt detected (Symmetry: H={sym['horizontal']}, V={sym['vertical']}). Please look directly at the camera."
        )
        
    # 2. SolvePnP Euler Angles (Secondary - handle 180-degree Pitch flip)
    # Pose is mathematically noisy; we only block on clear, extreme outliers (>45 deg)
    pitch, yaw = iris_data.head_pose['pitch'], iris_data.head_pose['yaw']
    
    # Check for 180-degree flip in pitch (common in SolvePnP/DecomposeProjectionMatrix)
    adjusted_pitch = abs(pitch)
    if adjusted_pitch > 90:
        adjusted_pitch = abs(adjusted_pitch - 180)
        
    if abs(yaw) > 35 or adjusted_pitch > 35:
        print(f"DEBUG: Pose Check Failed - Yaw: {yaw}, Adjusted Pitch: {adjusted_pitch}")
        raise HTTPException(
            status_code=422,
            detail=f"Head tilt detected (Pose: Y={round(yaw,1)}, P={round(adjusted_pitch,1)}). Please look directly at the camera."
        )
    
    # Parse reference type
    try:
        ref_type = ReferenceType(reference_type.lower())
    except ValueError:
        ref_type = ReferenceType.NONE
    
    # Calculate PD based on reference type
    if ref_type == ReferenceType.NONE:
        # Use iris estimation method
        pd_result = pd_calculator_service.calculate_pd_with_iris_estimation(iris_data)
    else:
        # Try to detect reference object
        reference_result = reference_detection_service.detect(cv_image, ref_type.value)
        
        if reference_result is None or not reference_result.detected:
            # Fall back to iris estimation if reference not detected
            pd_result = pd_calculator_service.calculate_pd_with_iris_estimation(iris_data)
            pd_result.reference_detected = False
            # Update disclaimer to indicate fallback
            pd_result.disclaimer = (
                f"⚠️ Reference object ({ref_type.value}) was not detected in the image. "
                f"Falling back to iris estimation method.\n\n{pd_result.disclaimer}"
            )
        else:
            # Use reference-based calculation
            pd_result = pd_calculator_service.calculate_pd_with_reference(iris_data, reference_result)
    
    # Convert to response model
    return PDMeasurementResult(
        overall_pd_mm=pd_result.overall_pd_mm,
        left_pd_mm=pd_result.left_pd_mm,
        right_pd_mm=pd_result.right_pd_mm,
        method=pd_result.method,
        model_used=pd_result.model_used,
        confidence_score=pd_result.confidence_score,
        error_margin=pd_result.error_margin,
        disclaimer=pd_result.disclaimer,
        face_detected=True,
        eyes_detected=True,
        reference_detected=pd_result.reference_detected,
        reference_scale_factor=pd_result.reference_scale_factor
    )


@app.get("/api/reference-types")
async def get_reference_types():
    """
    Get list of supported reference object types
    
    Returns information about each supported reference type including
    the real-world dimensions used for calibration.
    """
    return {
        "reference_types": [
            {
                "value": "none",
                "label": "No Reference (Iris Estimation)",
                "description": "Uses average human iris diameter (11.7mm) for scale estimation",
                "accuracy": "±1.5mm"
            },
            {
                "value": "credit_card",
                "label": "Credit Card",
                "description": "Standard ISO credit/debit card (85.6mm × 53.98mm)",
                "accuracy": "±1.0mm"
            },
            {
                "value": "coin_gbp_1p",
                "label": "1 Penny",
                "description": "UK 1 penny coin (20.3mm diameter)",
                "accuracy": "±1.0mm"
            },
            {
                "value": "coin_gbp_2p",
                "label": "2 Pence",
                "description": "UK 2 pence coin (25.9mm diameter)",
                "accuracy": "±1.0mm"
            },
            {
                "value": "coin_gbp_5p",
                "label": "5 Pence",
                "description": "UK 5 pence coin (18.0mm diameter)",
                "accuracy": "±1.0mm"
            },
            {
                "value": "coin_gbp_10p",
                "label": "10 Pence",
                "description": "UK 10 pence coin (24.5mm diameter)",
                "accuracy": "±1.0mm"
            },
            {
                "value": "coin_gbp_20p",
                "label": "20 Pence",
                "description": "UK 20 pence coin (21.4mm diameter, 7-sided)",
                "accuracy": "±1.0mm"
            },
            {
                "value": "coin_gbp_50p",
                "label": "50 Pence",
                "description": "UK 50 pence coin (27.3mm diameter, 7-sided)",
                "accuracy": "±1.0mm"
            },
            {
                "value": "coin_gbp_1",
                "label": "£1 Coin",
                "description": "UK £1 coin (23.43mm diameter)",
                "accuracy": "±1.0mm"
            },
            {
                "value": "coin_gbp_2",
                "label": "£2 Coin",
                "description": "UK £2 coin (28.4mm diameter)",
                "accuracy": "±1.0mm"
            },
            {
                "value": "ruler",
                "label": "Ruler",
                "description": "Standard ruler with 10mm segments",
                "accuracy": "±1.0mm"
            }
        ]
    }


# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    face_detection_service.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
