# Suggestions & Improvements Report

**Project:** Antigravity PD Measurement System
**Date:** 2026-04-04

---

## Priority Matrix

| Priority | Focus | Items | Effort |
|---|---|---|---|
| **P0 — Critical Fixes** | Correctness bugs that produce wrong results | 6 | 1–2 days |
| **P1 — Security** | Vulnerabilities that expose the system to abuse | 5 | 1 day |
| **P2 — Feature Completion** | Documented features that don't work | 4 | 2–3 days |
| **P3 — Architecture** | Structural improvements for reliability and DX | 6 | 3–5 days |
| **P4 — Code Quality** | Maintainability, testing, and polish | 7 | 3–5 days |

---

## P0 — Critical Fixes (Must Fix)

### 1. Fix Double Yaw Perspective Correction

**Location:** `backend/services/pd_calculator.py:38-63`

**Problem:** Yaw correction applied to both reference (iris diameter) AND measurement (PD distance) — results in `1/cos²(yaw)` overcorrection.

**Fix:** Remove the second correction. Only correct the reference dimension:

```python
# Correct iris diameter for foreshortening
corrected_left_dia  = iris_data.left_iris_diameter_px / np.cos(yaw_rad)
corrected_right_dia = iris_data.right_iris_diameter_px / np.cos(yaw_rad)

# Derive mm_per_pixel from corrected reference
avg_iris_diameter_px = np.mean([corrected_left_dia, corrected_right_dia])
mm_per_pixel = IRIS_DIAMETER_MM / avg_iris_diameter_px

# Do NOT apply yaw correction again — it's already baked into mm_per_pixel
corrected_pd_mm = pd_pixels * mm_per_pixel
```

**Impact:** Fixes up to 22% measurement error at maximum allowed yaw angle.

---

### 2. Implement Real Monocular PD Calculation

**Location:** `backend/services/pd_calculator.py:67-68` and `backend/main.py:116-117`

**Problem:** Left/right PD always set to `overall_pd / 2`, ignoring actual facial asymmetry.

**Fix:** Use the nose bridge midpoint (landmark 168) as the monocular reference:

```python
# In face_detection.py — extract nose bridge midpoint
nose_bridge = landmarks[168]
nose_bridge_x = nose_bridge.x * image_width

# In pd_calculator.py — calculate true monocular PDs
left_iris_center_x = iris_data.left_iris_center_x
right_iris_center_x = iris_data.right_iris_center_x
nose_x = face_data.nose_bridge_x

left_pd_px  = abs(left_iris_center_x - nose_x)
right_pd_px = abs(right_iris_center_x - nose_x)

left_pd_mm  = left_pd_px * mm_per_pixel
right_pd_mm = right_pd_px * mm_per_pixel
```

Also update the batch endpoint to compute independent medians for left and right PDs:

```python
left_values = sorted([r.left_pd_mm for r in results])
right_values = sorted([r.right_pd_mm for r in results])
final_result.left_pd_mm = true_median(left_values)
final_result.right_pd_mm = true_median(right_values)
```

**Impact:** Enables clinically meaningful monocular PD values for progressive/high-index lenses.

---

### 3. Fix Batch Median Calculation

**Location:** `backend/main.py:111`

**Problem:** Uses upper-middle index instead of true median for even-length arrays.

**Fix:**

```python
import statistics

median_pd = statistics.median(pd_values)
# Or manually:
n = len(pd_values)
median_pd = (pd_values[(n - 1) // 2] + pd_values[n // 2]) / 2
```

---

### 4. Fix Upload Mode (Frontend)

**Location:** `frontend/src/components/PDMeasurer/index.tsx`

**Problem:** Upload mode never populates `captureBufferRef`, so `handleMultiFrameSubmit` sends empty FormData.

**Fix:** For upload mode, convert the selected file to a blob and add it to the buffer, or use the single-image endpoint:

```tsx
const handleUploadSubmit = async () => {
    if (!imageFile) return;

    setStep("processing");
    const formData = new FormData();
    formData.append("image", imageFile);

    // Use single-image endpoint for uploads
    const response = await fetch(`${apiEndpoint}`, {
        method: "POST",
        body: formData,
    });
    const result = await response.json();
    setResult(result);
    setStep("results");
};
```

Alternatively, for batch consistency:

```tsx
const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
        setImageFile(file);
        captureBufferRef.current = [file];  // Add file to capture buffer
        if (imagePreview) URL.revokeObjectURL(imagePreview);
        setImagePreview(URL.createObjectURL(file));
    }
};
```

---

### 5. Fix Widget Shadow DOM Styles

**Location:** `frontend/src/embed.tsx` and `frontend/vite.widget.config.ts`

**Problem:** `__PD_WIDGET_STYLES__` is never set, so the Shadow DOM has no component styles.

**Fix:** Use Vite's `?inline` CSS import to inject styles directly:

```tsx
// embed.tsx
import componentStyles from "./components/PDMeasurer/styles.css?inline";

// In the init function, after creating the shadow root:
const styleEl = document.createElement("style");
styleEl.textContent = componentStyles;
shadow.appendChild(styleEl);
```

Or update `vite.widget.config.ts` to use `cssCodeSplit: false` and inject all CSS into the IIFE bundle.

---

### 6. Fix Confidence Score Pitch Correction

**Location:** `backend/services/face_detection.py`

**Problem:** `_calculate_advanced_confidence` uses raw uncorrected pitch, while the validator in `main.py` applies a 180-degree correction.

**Fix:** Apply the same pitch correction in the confidence scorer:

```python
def _calculate_advanced_confidence(self, ...):
    adjusted_pitch = pose['pitch']
    if adjusted_pitch > 90:
        adjusted_pitch = adjusted_pitch - 180
    elif adjusted_pitch < -90:
        adjusted_pitch = adjusted_pitch + 180

    tilt_penalty = max(0, abs(adjusted_pitch) - 10) * 0.02
```

---

## P1 — Security Hardening

### 7. Add Upload File Size Limit

**Location:** `backend/main.py`

```python
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

async def validate_image(image: UploadFile) -> bytes:
    contents = await image.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 10MB limit")
    return contents
```

---

### 8. Add Magic-Byte Validation

**Location:** `backend/main.py`

```python
ALLOWED_MAGIC_BYTES = {
    b'\xff\xd8\xff': "image/jpeg",
    b'\x89PNG': "image/png",
}

def validate_image_format(contents: bytes) -> None:
    for magic, fmt in ALLOWED_MAGIC_BYTES.items():
        if contents[:len(magic)] == magic:
            return
    raise HTTPException(status_code=400, detail="Unsupported image format. Only JPEG and PNG are accepted.")
```

---

### 9. Fix CORS Configuration

**Location:** `backend/main.py:40`

```python
ALLOWED_ORIGINS = [
    "https://your-frontend.vercel.app",
    "http://localhost:3000",  # development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

---

### 10. Add Rate Limiting

**Location:** `backend/main.py`

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/pd/measure-batch")
@limiter.limit("10/minute")
async def measure_batch(request: Request, ...):
    ...
```

---

### 11. Rotate and Secure HF Token

- **Immediately rotate** the Hugging Face token at https://huggingface.co/settings/tokens
- Remove `.env.hf` from the repository (check if it was ever committed with `git log --all -- .env.hf`)
- Update `hf_push.sh` to use environment variables or `git credential store` instead of embedding tokens in URLs:

```bash
# Instead of embedding token in URL:
git remote set-url hf "https://huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE_NAME}"
# Use credential helper:
git -c credential.helper='!f() { echo "password=${HF_TOKEN}"; }; f' push hf main --force
```

---

## P2 — Feature Completion

### 12. Add Reference Object UI Selector

**Location:** `frontend/src/components/PDMeasurer/index.tsx`

The backend fully supports reference objects (credit card, coins, ruler) but the frontend has no UI for selecting them. Add a reference type selector:

```tsx
const [referenceType, setReferenceType] = useState<string>("none");

// In the capture step UI:
<div className="reference-selector">
    <label>Reference Object (optional):</label>
    <select value={referenceType} onChange={e => setReferenceType(e.target.value)}>
        <option value="none">No reference object</option>
        <option value="credit_card">Credit Card</option>
        <option value="us_quarter">US Quarter</option>
        <option value="ruler_cm">Ruler (cm)</option>
    </select>
</div>

// In handleMultiFrameSubmit:
formData.append("reference_type", referenceType);
```

---

### 13. Display Error Margin and Confidence in Results

**Location:** `frontend/src/components/PDMeasurer/index.tsx` — results step

The API returns `confidence_score` and `measurement_error_mm` but the UI doesn't display them:

```tsx
// In the results step:
<div className="pd-results">
    <div className="pd-value">Overall PD: {result.overall_pd_mm}mm</div>
    <div className="pd-value">Left PD: {result.left_pd_mm}mm</div>
    <div className="pd-value">Right PD: {result.right_pd_mm}mm</div>

    <div className="pd-confidence">
        Confidence: {(result.confidence_score * 100).toFixed(0)}%
    </div>
    <div className="pd-error-margin">
        Error margin: ±{result.measurement_error_mm.toFixed(1)}mm
    </div>
</div>
```

---

### 14. Implement `primaryColor` Theming

**Location:** `frontend/src/components/PDMeasurer/index.tsx`

The `primaryColor` prop is accepted but never applied:

```tsx
useEffect(() => {
    if (primaryColor && containerRef.current) {
        containerRef.current.style.setProperty("--primary-color", primaryColor);
    }
}, [primaryColor]);
```

---

### 15. Add `mediapipeBasePath` Configuration for Widget

**Location:** `frontend/src/components/PDMeasurer/index.tsx` and `frontend/src/embed.tsx`

The `locateFile` callback hardcodes `/mediapipe/face_mesh/` which breaks on third-party sites:

```tsx
// Add prop
interface PDMeasurerProps {
    mediapipeBasePath?: string;
    // ...
}

// Use in FaceMesh initialization
const faceMesh = new FaceMesh({
    locateFile: (file: string) =>
        `${mediapipeBasePath || "/mediapipe/face_mesh"}/${file}`,
});
```

---

## P3 — Architecture Improvements

### 16. Add `docker-compose.yml` for Local Development

```yaml
# docker-compose.yml
version: "3.8"

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./backend:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    environment:
      - VITE_API_BASE_URL=http://localhost:8000
    volumes:
      - ./frontend/src:/app/src
    depends_on:
      - backend
```

---

### 17. Eliminate Redundant Server-Side MediaPipe Pass

Instead of sending raw images, send the client-side MediaPipe landmarks + raw image (for verification):

**Option A — Trust client landmarks (lower latency):**
```tsx
// Frontend: Send landmarks instead of raw images
const payload = {
    landmarks: faceLandmarks,
    image_width: video.videoWidth,
    image_height: video.videoHeight,
};
```

**Option B — Send both, verify on server (higher trust):**
```tsx
// Frontend: Send image + landmarks
formData.append("image", blob);
formData.append("client_landmarks", JSON.stringify(faceLandmarks));
// Backend: Run its own MediaPipe, compare with client landmarks for fraud detection
```

---

### 18. Add CI/CD Pipeline

**Location:** `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r backend/requirements.txt pytest
      - run: cd backend && pytest

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - run: cd frontend && npm ci
      - run: cd frontend && npx tsc --noEmit
      - run: cd frontend && npx vitest run
```

---

### 19. Add React Error Boundary

**Location:** `frontend/src/components/PDMeasurer/index.tsx` or `frontend/src/embed.tsx`

```tsx
class PDErrorBoundary extends React.Component<
    { children: React.ReactNode },
    { hasError: boolean; error?: Error }
> {
    state = { hasError: false, error: undefined };

    static getDerivedStateFromError(error: Error) {
        return { hasError: true, error };
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="pd-error">
                    <p>Something went wrong with the PD measurement tool.</p>
                    <button onClick={() => this.setState({ hasError: false })}>
                        Try Again
                    </button>
                </div>
            );
        }
        return this.props.children;
    }
}
```

---

### 20. Add MediaPipe Null-Check and Fallback

**Location:** `frontend/src/components/PDMeasurer/index.tsx`

```tsx
useEffect(() => {
    const FaceMesh = (window as any).FaceMesh;
    const Camera = (window as any).Camera;

    if (!FaceMesh || !Camera) {
        setError(
            "Face detection libraries failed to load. " +
            "Please check your internet connection and reload the page."
        );
        return;
    }

    initializeMediaPipe(FaceMesh, Camera);
}, []);
```

---

### 21. Add Cancellable Capture Loop

**Location:** `frontend/src/components/PDMeasurer/index.tsx`

```tsx
const captureAbortRef = useRef<AbortController | null>(null);

const triggerAutoCapture = async () => {
    const abort = new AbortController();
    captureAbortRef.current = abort;

    for (let i = 0; i < 10; i++) {
        if (abort.signal.aborted || !isComponentMounted.current) return;

        const blob = await captureFrame();
        if (blob) captureBufferRef.current.push(blob);

        await new Promise((resolve) => setTimeout(resolve, 100));
    }

    if (!abort.signal.aborted) {
        handleMultiFrameSubmit();
    }
};

// In cleanup/unmount:
useEffect(() => {
    return () => {
        captureAbortRef.current?.abort();
    };
}, []);
```

---

## P4 — Code Quality & Polish

### 22. Remove `dist/` from Git Tracking

```bash
git rm -r --cached frontend/dist/
# Ensure frontend/dist/ is in .gitignore (already is)
git commit -m "Remove tracked dist/ artifacts from repository"
```

---

### 23. Add Backend Tests

**Location:** `backend/tests/`

```python
# backend/tests/test_pd_calculator.py
import pytest
from services.pd_calculator import PDCalculatorService

class TestPDCalculator:
    def setup_method(self):
        self.calc = PDCalculatorService()

    def test_zero_yaw_no_correction(self):
        """At 0 yaw, PD should equal raw pixel calculation."""
        result = self.calc.calculate_pd(iris_data, pose_yaw=0.0)
        expected = (iris_data.pd_pixels / iris_data.avg_iris_diameter_px) * 11.8
        assert abs(result.overall_pd_mm - expected) < 0.01

    def test_yaw_correction_factor(self):
        """At 20 yaw, correction should be ~1/cos(20) = 1.064x, not 1.133x."""
        result_0 = self.calc.calculate_pd(iris_data, pose_yaw=0.0)
        result_20 = self.calc.calculate_pd(iris_data, pose_yaw=20.0)
        ratio = result_20.overall_pd_mm / result_0.overall_pd_mm
        assert 1.05 < ratio < 1.08  # ~1.064, NOT ~1.133

    def test_monocular_pd_asymmetry(self):
        """Left and right PD should differ for asymmetric faces."""
        result = self.calc.calculate_pd(asymmetric_iris_data, pose_yaw=0.0)
        assert result.left_pd_mm != result.right_pd_mm

    def test_extreme_yaw_guarded(self):
        """Near-90-degree yaw should not cause division by zero."""
        with pytest.raises(ValueError):
            self.calc.calculate_pd(iris_data, pose_yaw=89.0)
```

---

### 24. Add Frontend Tests

**Location:** `frontend/src/components/PDMeasurer/__tests__/`

```bash
npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
```

```tsx
// PDMeasurer.test.tsx
import { render, screen } from "@testing-library/react";
import { PDMeasurer } from "../index";

describe("PDMeasurer", () => {
    it("renders capture step by default", () => {
        render(<PDMeasurer apiEndpoint="/api/pd/measure" />);
        expect(screen.getByText(/align your face/i)).toBeInTheDocument();
    });

    it("shows error when MediaPipe is not loaded", () => {
        delete (window as any).FaceMesh;
        render(<PDMeasurer apiEndpoint="/api/pd/measure" />);
        expect(screen.getByText(/failed to load/i)).toBeInTheDocument();
    });
});
```

---

### 25. Replace Deprecated `on_event` with Lifespan

**Location:** `backend/main.py`

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    face_service.close()

app = FastAPI(title="PD Measurement API", lifespan=lifespan)
```

---

### 26. Fix `np.int_` Deprecation

**Location:** `backend/services/reference_detection.py:107`

```python
# Before (deprecated):
box = np.int_(box)

# After:
box = box.astype(np.int32)
```

---

### 27. Add Exception Handling Around `solvePnP`

**Location:** `backend/services/face_detection.py`

```python
try:
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None  # or return a default pose with low confidence
except cv2.error as e:
    logger.warning(f"solvePnP failed: {e}")
    return None
```

---

### 28. Fix Object URL Memory Leak

**Location:** `frontend/src/components/PDMeasurer/index.tsx`

```tsx
const prevPreviewRef = useRef<string | null>(null);

const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
        // Revoke previous URL
        if (prevPreviewRef.current) {
            URL.revokeObjectURL(prevPreviewRef.current);
        }
        const newUrl = URL.createObjectURL(file);
        prevPreviewRef.current = newUrl;
        setImagePreview(newUrl);
        setImageFile(file);
    }
};

// Cleanup on unmount
useEffect(() => {
    return () => {
        if (prevPreviewRef.current) {
            URL.revokeObjectURL(prevPreviewRef.current);
        }
    };
}, []);
```

---

## Summary

### Quick Wins (< 1 hour each)

| # | Fix | Impact |
|---|---|---|
| 3 | Fix batch median (use `statistics.median`) | Correct |
| 8 | Add magic-byte validation | Secure |
| 9 | Fix CORS configuration | Secure |
| 22 | Remove `dist/` from git tracking | Clean |
| 25 | Replace deprecated `on_event` | Modern |
| 26 | Fix `np.int_` deprecation | Future-proof |
| 28 | Fix Object URL memory leak | Stable |

### High-Impact Fixes (1–4 hours each)

| # | Fix | Impact |
|---|---|---|
| 1 | Fix double yaw correction | **Up to 22% accuracy improvement** |
| 2 | Implement real monocular PD | **Clinically meaningful left/right PD** |
| 4 | Fix upload mode | **Restores broken feature** |
| 5 | Fix widget Shadow DOM styles | **Makes widget usable** |
| 7 | Add file size limit | **Prevents DoS** |
| 10 | Add rate limiting | **Prevents abuse** |

### Strategic Improvements (1–5 days)

| # | Improvement | Impact |
|---|---|---|
| 12 | Reference object UI | Higher accuracy option for users |
| 17 | Eliminate dual MediaPipe | 50% server compute reduction |
| 18 | CI/CD pipeline | Automated quality gates |
| 23–24 | Backend + frontend tests | Regression prevention |
| 16 | docker-compose | Developer onboarding |

---

## Recommended Implementation Order

```
Week 1: Critical Fixes
├── Day 1: Fix #1 (yaw correction) + #3 (median) + #6 (pitch correction)
├── Day 2: Fix #2 (monocular PD) + #4 (upload mode)
└── Day 3: Fix #5 (widget styles) + #7-#11 (security)

Week 2: Feature Completion + Architecture
├── Day 1: Fix #12 (reference UI) + #13 (display confidence)
├── Day 2: Fix #14 (theming) + #15 (mediapipe base path) + #16 (docker-compose)
└── Day 3: Fix #18 (CI/CD) + #19 (error boundary) + #20 (null-check)

Week 3: Quality & Testing
├── Day 1: Fix #22 (dist cleanup) + #25-#27 (deprecations + error handling)
├── Day 2: Fix #23 (backend tests)
└── Day 3: Fix #24 (frontend tests) + #28 (memory leak)
```
