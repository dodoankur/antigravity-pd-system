"""
PD Measurement API - FastAPI Backend
Provides endpoints for measuring pupil distance from face images
"""
import io
import os
import time
import logging
import cv2
import numpy as np
import statistics
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pillow_heif import register_heif_opener
from contextlib import asynccontextmanager

# Register HEIF/HEIC support with Pillow globally
register_heif_opener()

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from models.schemas import (
    ReferenceType,
    PDMeasurementResult,
    HealthResponse,
    ErrorResponse,
    MeasurementMethod,
    ErrorMargin,
)
from services.face_detection import FaceDetectionService
from services.reference_detection import ReferenceDetectionService
from services.pd_calculator import PDCalculatorService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_VERSION = "1.0.0"

# CORS origins — comma-separated in env var, fallback to localhost dev defaults
_CORS_ORIGINS_RAW = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
)
CORS_ORIGINS: List[str] = [o.strip() for o in _CORS_ORIGINS_RAW.split(",") if o.strip()]

# Optional API-key auth — set API_KEY env var to enable
API_KEY = os.getenv("API_KEY", "")

# Image processing limits
MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_DIM = 1920                        # Resize long edge to this before processing

# ---------------------------------------------------------------------------
# Service singletons
# ---------------------------------------------------------------------------
face_detection_service: FaceDetectionService | None = None
reference_detection_service: ReferenceDetectionService | None = None
pd_calculator_service: PDCalculatorService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_detection_service, reference_detection_service, pd_calculator_service
    logger.info("Initialising ML services…")
    face_detection_service = FaceDetectionService()
    reference_detection_service = ReferenceDetectionService()
    pd_calculator_service = PDCalculatorService()
    logger.info("All services ready.")
    yield
    if face_detection_service:
        face_detection_service.close()
        logger.info("FaceDetectionService closed.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PD Measurement API",
    description="API for measuring pupil distance from face images",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s  status=%d  %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# API-key dependency (no-op when API_KEY is not configured)
# ---------------------------------------------------------------------------
async def verify_api_key(request: Request):
    if not API_KEY:
        return  # Auth disabled
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ALLOWED_MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"RIFF": "image/webp",       # WebP: "RIFF????WEBP"
    b"\x00\x00\x00\x18ftyp": "image/heic",   # HEIC/HEIF ftyp box (most common)
    b"\x00\x00\x00\x1cftyp": "image/heic",   # HEIC/HEIF ftyp box (alternate size)
    b"\x00\x00\x00\x14ftyp": "image/heic",   # HEIC/HEIF ftyp box (alternate size)
    b"\x00\x00\x00\x20ftyp": "image/heic",   # HEIC/HEIF ftyp box (alternate size)
}


def _validate_image_format(contents: bytes) -> None:
    """Raise HTTPException if the file header doesn't match allowed types."""
    # HEIC/HEIF: ftyp box at bytes 4-8, box size varies (bytes 0-3)
    if len(contents) >= 12 and contents[4:8] == b"ftyp":
        return
    for magic in ALLOWED_MAGIC_BYTES:
        if contents[: len(magic)] == magic:
            # Extra check for WebP: bytes 8-12 must be "WEBP"
            if magic == b"RIFF" and contents[8:12] != b"WEBP":
                continue
            return
    raise HTTPException(
        status_code=400,
        detail="Invalid or unsupported image format. Accepted: JPEG, PNG, WebP, HEIC/HEIF.",
    )


def _resize_if_needed(cv_image: np.ndarray) -> np.ndarray:
    """Downscale image if either dimension exceeds MAX_DIM. Preserves aspect ratio."""
    h, w = cv_image.shape[:2]
    if max(h, w) <= MAX_DIM:
        return cv_image
    scale = MAX_DIM / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug("Resized image from %dx%d to %dx%d", w, h, new_w, new_h)
    return resized


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint — health check."""
    return _health_response()


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint. Reports 'degraded' if ML services failed to load."""
    return _health_response()


def _health_response() -> HealthResponse:
    if face_detection_service is None or pd_calculator_service is None:
        return HealthResponse(status="degraded", version=API_VERSION)
    return HealthResponse(status="healthy", version=API_VERSION)


@app.post("/api/convert/heif")
@limiter.limit("30/minute")
async def convert_heif(
    request: Request,
    image: UploadFile = File(..., description="HEIC or HEIF image file"),
) -> StreamingResponse:
    """
    Convert a HEIC/HEIF image to JPEG and return the JPEG bytes.
    Used by the frontend to get a browser-renderable version for preview
    and to send a decodable file to the measurement endpoint.
    """
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        pil_img = Image.open(io.BytesIO(data))
        # Convert to RGB — HEIF can be RGBA or other modes JPEG doesn't support
        if pil_img.mode not in ("RGB", "L"):
            pil_img = pil_img.convert("RGB")

        out = io.BytesIO()
        pil_img.save(out, format="JPEG", quality=92)
        out.seek(0)

        logger.info("HEIF conversion OK: %s -> JPEG (%d bytes)", image.filename, out.getbuffer().nbytes)
        return StreamingResponse(
            out,
            media_type="image/jpeg",
            headers={"Content-Disposition": f'inline; filename="{image.filename or "converted"}.jpg"'},
        )
    except Exception as e:
        logger.error("HEIF conversion failed: %s", e)
        raise HTTPException(status_code=422, detail=f"Could not convert HEIF image: {e}")


@app.post("/api/pd/measure", response_model=PDMeasurementResult)
@limiter.limit("10/minute")
async def measure_pd(
    request: Request,
    image: UploadFile = File(..., description="Face image with open eyes"),
    reference_type: str = Form(default="none", description="Type of reference object in image"),
    age_group: str = Form(default="auto", description="Age group: auto, adult, teen, child, young_child"),
    _auth=Depends(verify_api_key),
):
    """Measure pupil distance from a single uploaded face image."""
    return await _process_single_image(image, reference_type, age_group)


@app.post("/api/pd/diagnose-frame")
async def diagnose_frame(
    image: UploadFile = File(...),
):
    """DIAGNOSTIC: returns blur, brightness, face-detected, and rejection reason for one frame.
    No auth required — REMOVE before production.
    """
    import io as _io
    import numpy as _np
    import cv2 as _cv2
    from PIL import Image as _Image

    contents = await image.read()
    try:
        pil_image = _Image.open(_io.BytesIO(contents))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        cv_image = _cv2.cvtColor(_np.array(pil_image), _cv2.COLOR_RGB2BGR)
    except Exception as e:
        return {"error": f"Decode failed: {e}"}

    h, w = cv_image.shape[:2]
    img_size = f"{w}x{h}"   # capture before any local variable shadows h/w
    quality = face_detection_service.assess_quality(cv_image, camera_mode=True)

    iris = face_detection_service.detect_iris(cv_image, camera_mode=True)
    face_detected = iris is not None
    reject_reason = iris.reject_reason if iris and not iris.is_usable else None
    is_usable     = iris.is_usable     if iris else False
    yaw   = iris.head_pose.get("yaw")   if iris else None
    pitch = iris.head_pose.get("pitch") if iris else None
    roll  = iris.head_pose.get("roll")  if iris else None
    sym_h = iris.pose_symmetry.get("horizontal") if iris else None
    sym_v = iris.pose_symmetry.get("vertical")   if iris else None

    # Simulate the _process_single_image symmetry + pose gates exactly
    gate_sym_fail  = None
    gate_pose_fail = None
    if iris:
        sym = iris.pose_symmetry
        sh, sv = sym.get("horizontal", 0.5), sym.get("vertical", 0.5)
        if not (0.35 <= sh <= 0.65) or not (0.30 <= sv <= 0.70):
            gate_sym_fail = f"H={sh:.3f} V={sv:.3f} (need H:0.35-0.65 V:0.30-0.70)"
        adj_p = abs(pitch or 0)
        if adj_p > 90: adj_p = abs(adj_p - 180)
        if abs(yaw or 0) > 35 or adj_p > 35:
            gate_pose_fail = f"yaw={yaw:.1f} adj_pitch={adj_p:.1f} (need <35)"

    return {
        "image_size":       img_size,
        "blur_score":       quality["blur"],
        "brightness":       quality["brightness"],
        "is_reliable":      quality["is_reliable"],
        "MIN_BLUR_CAMERA":  face_detection_service.MIN_BLUR_CAMERA,
        "MIN_BLUR":         face_detection_service.MIN_BLUR,
        "face_detected":    face_detected,
        "is_usable":        is_usable,
        "reject_reason":    reject_reason,
        "yaw_deg":          yaw,
        "pitch_deg":        pitch,
        "pitch_adj_deg":    round(abs(abs(pitch or 0) - 180), 2) if pitch is not None and abs(pitch or 0) > 90 else round(abs(pitch or 0), 2),
        "roll_deg":         roll,
        "sym_horizontal":   sym_h,
        "sym_vertical":     sym_v,
        "gate_sym_fail":    gate_sym_fail,
        "gate_pose_fail":   gate_pose_fail,
        "note_pitch":       "pitch≈±180° is normal for frontal face (solvePnP convention). adj_pitch_deg is the true tilt." if pitch is not None and abs(pitch) > 90 else None,
    }


@app.post("/api/pd/measure-batch", response_model=PDMeasurementResult)
@limiter.limit("10/minute")
async def measure_pd_batch(
    request: Request,
    images: List[UploadFile] = File(..., description="List of face images for median calculation"),
    reference_type: str = Form(default="none", description="Type of reference object in image"),
    age_group: str = Form(default="auto", description="Age group: auto, adult, teen, child, young_child"),
    _auth=Depends(verify_api_key),
):
    """
    Measure PD from multiple images and return a stable result using:
      1. Frame gating  — bad frames (blink/blur/yaw) are tracked and skipped
      2. IQR filtering — outlier frames outside 1.5×IQR are dropped
      3. Confidence-weighted median — high-confidence frames steer the result
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")

    results: list[PDMeasurementResult] = []
    rejected_count = 0
    errors = []
    # BUG A pass-1: collect age-group votes from each frame using per-frame
    # auto-detection.  We do NOT use these PD values — they may have been
    # computed with inconsistent iris_diameter_mm (e.g. 7 adult + 3 teen
    # frames).  We only care about the age_group_used field.
    age_group_votes: list[str] = []

    for img in images:
        try:
            res = await _process_single_image(img, reference_type, age_group, camera_mode=True)
            if res.age_group_used:
                age_group_votes.append(res.age_group_used)
        except HTTPException as e:
            detail = str(e.detail)
            logger.info("Batch pass-1 frame rejected: %s", detail)
            if any(k in detail for k in ("blink", "blur", "yaw", "tilt", "Symmetry", "roll")):
                rejected_count += 1
            else:
                errors.append(detail)
        except Exception:
            logger.exception("Unexpected error in batch pass-1 (age vote)")

    # --- BUG A: majority-vote age group before the real PD pass ---
    # Resolve consensus now so every frame in pass-2 uses the same diameter.
    if age_group == "auto" and age_group_votes:
        from collections import Counter
        vote_counts  = Counter(age_group_votes)
        majority_age = vote_counts.most_common(1)[0][0]
        logger.info(
            "Batch age-group majority vote: %s (votes: %s)",
            majority_age,
            dict(vote_counts),
        )
    else:
        majority_age = age_group if age_group != "auto" else "adult"

    # Resolve consensus iris_diameter_mm from majority age
    from services.pd_calculator import IRIS_DIAMETER_BY_AGE
    consensus_diameter_mm = IRIS_DIAMETER_BY_AGE.get(majority_age, IRIS_DIAMETER_BY_AGE["adult"])
    logger.info(
        "Batch pass-2: all frames will use iris_diameter_mm=%.2f (age=%s)",
        consensus_diameter_mm,
        majority_age,
    )

    # BUG A pass-2: re-process every image with the consensus diameter so all
    # frame PD values share the same mm/px scale — making the weighted mean
    # internally consistent.  Frame read position is reset by re-seeking.
    for img in images:
        try:
            await img.seek(0)
            res = await _process_single_image(
                img, reference_type, majority_age,
                force_iris_diameter_mm=consensus_diameter_mm,
                camera_mode=True,
            )
            results.append(res)
        except HTTPException as e:
            detail = str(e.detail)
            logger.info("Batch pass-2 frame rejected: %s", detail)
            if any(k in detail for k in ("blink", "blur", "yaw", "tilt", "Symmetry", "roll")):
                rejected_count += 1
            else:
                errors.append(detail)
        except Exception:
            logger.exception("Unexpected error processing batch frame (pass-2)")
            errors.append("Internal processing error")

    if not results:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Failed to process any images in the batch. "
                f"Rejected (quality): {rejected_count}. "
                f"Errors: {'; '.join(errors[:3])}"
            ),
        )

    # --- IQR outlier rejection (now absolute ±2mm, T2-C) ---
    assert pd_calculator_service is not None
    pd_values = [r.overall_pd_mm for r in results]
    keep_idx  = pd_calculator_service.iqr_filter(pd_values)
    if len(keep_idx) < len(results):
        logger.info(
            "IQR filter: dropped %d/%d frames",
            len(results) - len(keep_idx), len(results),
        )
    good_results = [results[i] for i in keep_idx]

    # --- T2-B: confidence-weighted MEAN of all kept frames ---
    # Previously we picked the single frame at the weighted-median index,
    # discarding the other 9 frames. A weighted mean uses all accepted
    # frames and reduces variance by ~sqrt(N) — roughly 0.5–1mm improvement
    # for a 10-frame session.
    good_weights = [r.confidence_score for r in good_results]
    total_w      = sum(good_weights) or float(len(good_results))
    norm_w       = [w / total_w for w in good_weights]

    mean_overall = round(sum(r.overall_pd_mm * w for r, w in zip(good_results, norm_w)), 1)
    mean_left    = round(sum(r.left_pd_mm    * w for r, w in zip(good_results, norm_w)), 1)
    mean_right   = round(sum(r.right_pd_mm   * w for r, w in zip(good_results, norm_w)), 1)
    mean_conf    = round(sum(r.confidence_score * w for r, w in zip(good_results, norm_w)), 3)

    # Base metadata from the highest-confidence kept frame
    # (error_margin, asymmetry_warning, method all come from that frame)
    best_frame = max(good_results, key=lambda r: r.confidence_score)

    frames_accepted       = len(good_results)
    frames_total          = len(images)
    frames_rejected_total = frames_total - frames_accepted  # includes quality + IQR

    # IMP E: base error margin from confidence tier, then scale by sqrt(10/N).
    # With N=10 good frames the scale factor is 1.0 (no change).
    # With N=3 accepted frames it widens by sqrt(10/3) ≈ 1.83 — more honest
    # about the higher variance from fewer observations.  Cap at 3.0mm.
    _base_error = 1.0 if mean_conf > 0.9 else (1.5 if mean_conf > 0.7 else 2.5)
    _n_scale    = float(np.sqrt(10.0 / max(frames_accepted, 1)))
    mean_error_val = min(round(_base_error * _n_scale, 2), 3.0)

    logger.info(
        "Batch result: overall=%.1f left=%.1f right=%.1f  "
        "accepted=%d/%d  conf=%.3f  asym_warn=%s  age=%s",
        mean_overall, mean_left, mean_right,
        frames_accepted, frames_total,
        mean_conf,
        best_frame.asymmetry_warning,
        majority_age,
    )

    return PDMeasurementResult(
        overall_pd_mm=mean_overall,
        left_pd_mm=mean_left,
        right_pd_mm=mean_right,
        method=best_frame.method,
        model_used=(
            f"Batch Weighted Mean ({frames_accepted}/{frames_total} frames, "
            f"±2mm filter+quality gated)"
        ),
        confidence_score=mean_conf,
        error_margin=ErrorMargin(
            value_mm=mean_error_val,
            percentage=round((mean_error_val / mean_overall) * 100, 1) if mean_overall > 0 else 0.0,
            confidence_score=mean_conf,
        ),
        disclaimer=(
            f"Calculated using {frames_accepted}-frame weighted-mean batch "
            f"(±2mm outlier filter + blink/blur/yaw gating). "
            + best_frame.disclaimer
        ),
        face_detected=True,
        eyes_detected=True,
        reference_detected=best_frame.reference_detected,
        reference_scale_factor=best_frame.reference_scale_factor,
        asymmetry_warning=best_frame.asymmetry_warning,
        age_group_used=majority_age,
        frames_accepted=frames_accepted,
        frames_rejected=frames_rejected_total,
    )


async def _process_single_image(
    image: UploadFile,
    reference_type: str,
    age_group: str = "auto",
    force_iris_diameter_mm: float = 0.0,
    camera_mode: bool = False,
) -> PDMeasurementResult:
    """Process a single image and return PD result.

    Args:
        image:                 Uploaded image file.
        reference_type:        "none", "card", etc.
        age_group:             "auto" or one of adult/teen/child/young_child.
        force_iris_diameter_mm: When > 0, skip per-frame age auto-detection and
                               use this diameter directly.  Used by the batch
                               endpoint (BUG A fix) to ensure all frames in a
                               batch share a single consensus iris_diameter_mm
                               rather than blending frames computed at different
                               diameters (e.g. 7 adult + 3 teen frames).
        camera_mode:           When True, uses MIN_BLUR_CAMERA (55) instead of
                               MIN_BLUR (80) for blur gating. Camera frames from
                               Safari/MediaPipe→canvas→JPEG are inherently softer
                               than static upload photos.
    """
    from services.pd_calculator import IRIS_DIAMETER_BY_AGE
    t0 = time.perf_counter()

    try:
        contents = await image.read()

        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Image exceeds 10MB limit.")

        _validate_image_format(contents)

        pil_image = Image.open(io.BytesIO(contents))

        # Normalise colour space
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to decode uploaded image")
        raise HTTPException(
            status_code=400,
            detail="Failed to process image. Please try a different photo.",
        )

    # Resize large images before running MediaPipe (major latency saving)
    cv_image = _resize_if_needed(cv_image)

    # Face & iris detection
    iris_data = face_detection_service.detect_iris(cv_image, camera_mode=camera_mode)
    if iris_data is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "No face detected in image. "
                "Please upload a clear, front-facing photo with both eyes visible."
            ),
        )

    # Inject age-group calibrated iris diameter (T3-3A / T3-3B)
    # BUG A fix: if the batch endpoint resolved a consensus diameter already,
    # use it directly — skip per-frame auto-detection to avoid blending frames
    # that were computed with different iris_diameter_mm values.
    if force_iris_diameter_mm > 0.0:
        iris_data.iris_diameter_mm = force_iris_diameter_mm
        resolved_age = age_group if age_group != "auto" else "adult"  # label only
        logger.debug("Force iris_diameter_mm=%.2f (consensus from batch)", force_iris_diameter_mm)
    elif age_group == "auto":
        detected = iris_data.estimated_age_group
        resolved_age = detected if detected != "unknown" else "adult"
        logger.info("Auto age-detection: estimated=%s resolved=%s", detected, resolved_age)
        iris_data.iris_diameter_mm = IRIS_DIAMETER_BY_AGE.get(resolved_age, IRIS_DIAMETER_BY_AGE["adult"])
    else:
        resolved_age = age_group
        iris_data.iris_diameter_mm = IRIS_DIAMETER_BY_AGE.get(resolved_age, IRIS_DIAMETER_BY_AGE["adult"])

    # Frame quality gate: face_detection now returns is_usable=False with a reason
    # instead of returning None, so batch callers can track rejection counts.
    if not iris_data.is_usable:
        raise HTTPException(
            status_code=422,
            detail=f"Frame rejected ({iris_data.reject_reason}). Please keep eyes open and face the camera.",
        )

    # Pose / symmetry validation
    sym = iris_data.pose_symmetry
    if not (0.35 <= sym["horizontal"] <= 0.65) or not (0.30 <= sym["vertical"] <= 0.70):
        logger.debug(
            "Symmetry check failed — H=%.3f V=%.3f",
            sym["horizontal"],
            sym["vertical"],
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"Head tilt detected (Symmetry: H={sym['horizontal']:.2f}, "
                f"V={sym['vertical']:.2f}). Please look directly at the camera."
            ),
        )

    pitch = iris_data.head_pose["pitch"]
    yaw = iris_data.head_pose["yaw"]
    adjusted_pitch = abs(pitch)
    if adjusted_pitch > 90:
        adjusted_pitch = abs(adjusted_pitch - 180)

    if abs(yaw) > 35 or adjusted_pitch > 35:
        logger.debug("Pose check failed — yaw=%.1f adjusted_pitch=%.1f", yaw, adjusted_pitch)
        raise HTTPException(
            status_code=422,
            detail=(
                f"Head tilt detected (Pose: Y={yaw:.1f}, P={adjusted_pitch:.1f}). "
                "Please look directly at the camera."
            ),
        )

    # Parse reference type — guard against None / empty / unknown values
    ref_type_str = (reference_type or "none").strip().lower()
    try:
        ref_type = ReferenceType(ref_type_str)
    except ValueError:
        ref_type = ReferenceType.NONE

    # Calculate PD
    if ref_type == ReferenceType.NONE:
        pd_result = pd_calculator_service.calculate_pd_with_iris_estimation(iris_data)
    else:
        reference_result = reference_detection_service.detect(cv_image, ref_type.value)
        if reference_result is None or not reference_result.detected:
            pd_result = pd_calculator_service.calculate_pd_with_iris_estimation(iris_data)
            pd_result.reference_detected = False
            pd_result.disclaimer = (
                f"Reference object ({ref_type.value}) was not detected. "
                f"Falling back to iris estimation.\n\n{pd_result.disclaimer}"
            )
        else:
            pd_result = pd_calculator_service.calculate_pd_with_reference(iris_data, reference_result)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "PD measured: overall=%.1f  method=%s  conf=%.2f  %.1fms",
        pd_result.overall_pd_mm,
        pd_result.method,
        pd_result.confidence_score,
        elapsed_ms,
    )

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
        reference_scale_factor=pd_result.reference_scale_factor,
        asymmetry_warning=pd_result.asymmetry_warning,
        age_group_used=resolved_age,
    )


@app.get("/api/reference-types")
async def get_reference_types():
    """Get list of supported reference object types."""
    return {
        "reference_types": [
            {
                "value": "none",
                "label": "No Reference (Iris Estimation)",
                "description": "Uses average human iris diameter (11.7mm) for scale estimation",
                "accuracy": "±1.5mm",
            },
            {
                "value": "credit_card",
                "label": "Credit Card",
                "description": "Standard ISO credit/debit card (85.6mm × 53.98mm)",
                "accuracy": "±1.0mm",
            },
            {
                "value": "coin_gbp_1p",
                "label": "1 Penny",
                "description": "UK 1 penny coin (20.3mm diameter)",
                "accuracy": "±1.0mm",
            },
            {
                "value": "coin_gbp_2p",
                "label": "2 Pence",
                "description": "UK 2 pence coin (25.9mm diameter)",
                "accuracy": "±1.0mm",
            },
            {
                "value": "coin_gbp_5p",
                "label": "5 Pence",
                "description": "UK 5 pence coin (18.0mm diameter)",
                "accuracy": "±1.0mm",
            },
            {
                "value": "coin_gbp_10p",
                "label": "10 Pence",
                "description": "UK 10 pence coin (24.5mm diameter)",
                "accuracy": "±1.0mm",
            },
            {
                "value": "coin_gbp_20p",
                "label": "20 Pence",
                "description": "UK 20 pence coin (21.4mm diameter, 7-sided)",
                "accuracy": "±1.0mm",
            },
            {
                "value": "coin_gbp_50p",
                "label": "50 Pence",
                "description": "UK 50 pence coin (27.3mm diameter, 7-sided)",
                "accuracy": "±1.0mm",
            },
            {
                "value": "coin_gbp_1",
                "label": "£1 Coin",
                "description": "UK £1 coin (23.43mm diameter)",
                "accuracy": "±1.0mm",
            },
            {
                "value": "coin_gbp_2",
                "label": "£2 Coin",
                "description": "UK £2 coin (28.4mm diameter)",
                "accuracy": "±1.0mm",
            },
            {
                "value": "ruler",
                "label": "Ruler",
                "description": "Standard ruler with 10mm segments",
                "accuracy": "±1.0mm",
            },
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
