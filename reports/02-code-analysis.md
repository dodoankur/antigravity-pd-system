# Code Analysis Report

**Project:** Antigravity PD Measurement System
**Date:** 2026-04-04

---

## 1. Overview

| Metric | Value |
|---|---|
| Total source files | ~15 (excluding configs, docs, built artifacts) |
| Backend LOC | ~650 (Python) |
| Frontend LOC | ~550 (TypeScript/TSX) |
| Test files | 0 |
| CI/CD configs | 0 |
| Documentation files | 17 |

---

## 2. Backend Code Analysis

### 2.1 `main.py` — FastAPI Application (150 lines)

**Responsibilities:** App initialization, CORS, endpoint definitions, batch processing logic.

#### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Problem:** `allow_origins=["*"]` combined with `allow_credentials=True` is an RFC violation. Browsers will block credentialed cross-origin requests when the origin is a wildcard. This silently breaks any cookie/header-based auth that might be added later, while giving a false sense of open access.

#### Batch Median Calculation

```python
pd_values = sorted([r.overall_pd_mm for r in results])
median_pd = pd_values[len(pd_values) // 2]
```

**Problem:** For even-length arrays (e.g., 10 results), this selects the upper-middle value (index 5 = 6th element), not the true median. A proper median is `(pd_values[4] + pd_values[5]) / 2` for 10 elements.

#### Monocular PD Override

```python
final_result.left_pd_mm = median_pd / 2
final_result.right_pd_mm = median_pd / 2
```

**Problem:** This **overwrites** independently calculated monocular PDs with an exact 50/50 split. Every patient gets perfectly symmetrical PD values regardless of actual facial anatomy. This is clinically incorrect — real monocular PDs differ by 0.5–2mm in most people.

#### Deprecated Lifecycle Event

```python
@app.on_event("shutdown")
async def shutdown_event():
    face_service.close()
```

**Problem:** `@app.on_event()` is deprecated in FastAPI. Should use the `lifespan` context manager pattern.

#### Missing Input Validation

- No maximum file size on `UploadFile`
- Content-type check only verifies `startswith("image/")` — allows SVG, WebP, BMP, TIFF, etc.
- No magic-byte validation to prevent content-type spoofing
- No rate limiting on any endpoint

---

### 2.2 `services/face_detection.py` — FaceDetectionService (200 lines)

**Responsibilities:** MediaPipe FaceMesh initialization, face landmark extraction, head pose estimation, iris measurement, confidence scoring.

#### Positive Aspects

- Uses `refine_landmarks=True` for the 478-point model (includes 10 iris landmarks)
- Iris diameter computed as median of 4 edge distances — robust against single-landmark jitter
- Confidence scoring uses weighted multi-factor approach (iris clarity, symmetry, pose, face size)

#### `decomposeProjectionMatrix` Fallback Error

```python
decomp = cv2.decomposeProjectionMatrix(proj_matrix)
if len(decomp) >= 7:
    euler_angles = decomp[6]
else:
    euler_angles = decomp[1]  # WRONG: index 1 is rotMatrix, not euler angles
```

**Problem:** `cv2.decomposeProjectionMatrix` always returns exactly 7 values in all modern OpenCV versions. The fallback path (index 1 = rotation matrix, a 3x3 matrix) would be interpreted as euler angles (expected 3 values), producing completely wrong pitch/yaw/roll. This would silently pass tilted/rotated faces through pose validation. While the fallback is unlikely to trigger on modern OpenCV, it's a dangerous silent-failure mode.

#### Missing Exception Handling Around `solvePnP`

```python
success, rotation_vector, translation_vector = cv2.solvePnP(
    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
)
```

**Problem:** No `try/except` wrapper. If landmarks are degenerate (coplanar, insufficient, or NaN), `solvePnP` can raise exceptions or return `success=False`. The code does check `success` later but doesn't handle the exception case.

#### Misleading Field Name

```python
face_width_px = distance(landmarks[LEFT_EYE_OUTER], landmarks[LEFT_EYE_INNER])
```

**Problem:** `face_width_px` actually stores the horizontal span of the **left eye only** (canthus-to-canthus distance, ~28–32mm). The name suggests it's the full face width (~140mm). This is used downstream in `pd_calculator.py` as a reference anchor of `30.0mm`, which is reasonable for a single eye — but the naming is deceptive and error-prone for future maintainers.

#### Confidence Score Not Pose-Corrected

```python
def _calculate_advanced_confidence(self, ...):
    # Uses raw pose['pitch'] directly
    tilt_penalty = max(0, abs(pose['pitch']) - 10) * 0.02
```

**Problem:** In `main.py`, a 180-degree flip correction is applied to pitch values (`adjusted_pitch = pitch - 180 if pitch > 90`). But the confidence scorer uses the raw uncorrected pitch. A face with `pitch=170` (which maps to `adjusted_pitch=10` in the validator — acceptable) would still receive a heavy tilt penalty of `(170 - 10) * 0.02 = 3.2` in the confidence function, driving confidence to 0.

---

### 2.3 `services/pd_calculator.py` — PDCalculatorService (100 lines)

**Responsibilities:** PD calculation from iris landmarks, perspective correction, reference-object-based calculation.

#### Double Yaw Correction (Critical Bug)

```python
# Step 1: Correct iris diameter for yaw
corrected_left_dia  = iris_data.left_iris_diameter_px / np.cos(yaw_rad)
corrected_right_dia = iris_data.right_iris_diameter_px / np.cos(yaw_rad)

# Step 2: Derive mm_per_pixel from corrected diameter
avg_iris_diameter_px = np.mean([corrected_left_dia, corrected_right_dia])
mm_per_pixel = 11.8 / avg_iris_diameter_px

# Step 3: ALSO correct the PD measurement for yaw
corrected_pd_mm = (pd_pixels / np.cos(yaw_rad)) * mm_per_pixel
```

**Problem:** The yaw correction is applied **twice** — once to the reference dimension (iris diameter in step 1) and once to the measurement (PD pixels in step 3). This results in an overcorrection of `1/cos^2(yaw)` instead of the correct `1/cos(yaw)`.

**Impact at various yaw angles:**

| Yaw | cos(yaw) | Correct factor | Actual factor | Error |
|---|---|---|---|---|
| 5° | 0.996 | 1.004x | 1.008x | +0.4% |
| 15° | 0.966 | 1.035x | 1.072x | +3.5% |
| 25° | 0.906 | 1.103x | 1.217x | +10.3% |
| 35° | 0.819 | 1.221x | 1.490x | +22.0% |

At 35° yaw (the maximum allowed), PD is overcorrected by **22%** — a 63mm PD would be reported as ~77mm.

#### Monocular PD Always 50/50

```python
left_pd_mm = corrected_pd_mm / 2
right_pd_mm = corrected_pd_mm / 2
```

**Problem:** This is a known placeholder. The proper approach is to compute each monocular PD as the distance from each iris center to the nose bridge midpoint (landmark 168). The code has access to all iris center coordinates but doesn't use them for monocular calculation.

#### Division by Zero Risk

```python
corrected_pd_mm = (pd_pixels / np.cos(yaw_rad)) * mm_per_pixel
```

**Problem:** For `yaw_rad` near `±π/2`, `cos(yaw_rad)` approaches 0. The pose validator in `main.py` gates at ±35° (where `cos(35°) ≈ 0.82`), so this is protected in practice. However, the calculator itself has no defensive check — if called directly or if the validator threshold is relaxed, division by zero occurs silently (NumPy returns `inf`).

---

### 2.4 `services/reference_detection.py` — ReferenceDetectionService (180 lines)

**Responsibilities:** Detect credit cards, coins, or rulers in the image for calibrated PD measurement.

#### Coin Detection Ignores Coin Size

```python
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
    minDist=50, param1=50, param2=30,
    minRadius=20, maxRadius=200)
```

**Problem:** The `minRadius`/`maxRadius` parameters are fixed regardless of which coin type is being detected. A 5p coin (18mm diameter) and a £2 coin (28.4mm diameter) at the same distance from the camera would produce different pixel radii, but the detection parameters are identical. The code detects **any** circle in the 20–200 pixel radius range and takes the best-scoring one, regardless of whether it matches the expected coin size.

#### Credit Card Confidence is Inflated

```python
confidence = 1.0 - (ratio_diff / self.card_aspect_ratio)
```

**Problem:** With a 15% aspect ratio tolerance, `ratio_diff` ranges from 0 to `0.238`. So confidence ranges from `0.85` to `1.0` — a narrow band where "low confidence" (0.85) is virtually indistinguishable from "high confidence" (1.0). This makes the confidence score meaningless for downstream decision-making.

#### Ruler Detection False Positives

The ruler detector looks for evenly-spaced vertical lines but has no size or context constraints. Any image containing regularly-spaced vertical features (window blinds, striped fabric, bookshelf edges, keyboard keys) could trigger a false positive detection.

#### Deprecated NumPy Call

```python
box = np.int_(box)  # line 107
```

**Problem:** `np.int_()` was deprecated in NumPy 1.24 and will raise an error in future versions. Should be `box.astype(np.int32)`.

---

## 3. Frontend Code Analysis

### 3.1 `components/PDMeasurer/index.tsx` — Main Component (437 lines)

**Responsibilities:** Camera management, real-time face validation, frame capture, API communication, results display.

#### MediaPipe Global Dependency

```tsx
const FaceMesh = (window as any).FaceMesh;
const Camera = (window as any).Camera;
```

**Problem:** MediaPipe is loaded via `<script>` tags and accessed as untyped globals. There is no null-check before `new FaceMesh(...)` — if the CDN fails to load, the constructor call will throw `TypeError: FaceMesh is not a constructor` with no user-facing error message. TypeScript's type safety is completely bypassed via `as any`.

#### Upload Mode is Broken

The component has two input modes:
1. **Camera mode** — auto-captures 10 frames to `captureBufferRef`, then calls `handleMultiFrameSubmit`
2. **Upload mode** — user selects a file, clicks "Measure PD", calls `handleMultiFrameSubmit`

```tsx
// handleMultiFrameSubmit sends captureBufferRef contents
captureBufferRef.current.forEach((blob, index) => {
    formData.append("images", blob, `frame_${index}.jpg`);
});
```

**Problem:** In upload mode, the selected file is stored in `imageFile` state but **never added to `captureBufferRef`**. When `handleMultiFrameSubmit` is called, it sends an empty `FormData` (zero images). The backend will reject this with "No images provided".

#### Non-Cancellable Auto-Capture Loop

```tsx
const triggerAutoCapture = async () => {
    for (let i = 0; i < 10; i++) {
        // capture frame to blob
        captureBufferRef.current.push(blob);
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    handleMultiFrameSubmit();
};
```

**Problem:** This loop runs for ~1 second with no cancellation mechanism. If the user navigates away, switches tabs, or the component unmounts mid-capture, the loop continues executing. While `isComponentMounted.current` exists, it's not checked inside the loop.

#### Fragile URL Construction

```tsx
const batchEndpoint = apiEndpoint.includes("/api/pd/measure")
    ? apiEndpoint.replace("/api/pd/measure", "/api/pd/measure-batch")
    : `${apiEndpoint.replace(/\/$/, "")}/batch`;
```

**Problem:** `String.replace()` replaces the **first occurrence** of the substring. If `apiEndpoint` is `https://api.example.com/api/pd/measure-v2`, the result would be `https://api.example.com/api/pd/measure-batch-v2` — an invalid URL. The fallback branch appends `/batch` rather than the correct `/api/pd/measure-batch`.

#### Memory Leak — Object URLs

```tsx
setImagePreview(URL.createObjectURL(file));
```

**Problem:** Each file selection creates a new Object URL, but the previous URL is never revoked via `URL.revokeObjectURL()`. Over multiple file selections, this leaks memory.

#### Stale Closure Risk

```tsx
const onResults = useCallback((results) => {
    // calls triggerAutoCapture()
}, []);  // empty dependency array
```

`triggerAutoCapture` is a regular `async function` defined inline (not memoized). Since `onResults` captures it at creation time (once, due to `[]` deps), the `triggerAutoCapture` reference is always from the first render. This is a stale closure — if `triggerAutoCapture` ever needs to read props or state that changes, it will read stale values.

#### Reset/Camera Race Condition

```tsx
const handleReset = () => {
    // ... state resets ...
    startCamera();
};
```

**Problem:** `startCamera` has an `initializationLock` guard. If the camera is still tearing down when `handleReset` is called, `startCamera` will immediately return (lock is held), and the camera will never reinitialize.

#### Unused Props

- `primaryColor` — destructured from props but never applied to any CSS variable or style
- No reference type selector — `reference_type` is always hardcoded to `"none"`

---

### 3.2 `embed.tsx` — Widget Entry Point (160 lines)

**Responsibilities:** Shadow DOM encapsulation, global API (`window.PDWidget`), lifecycle management.

#### Shadow DOM Styles Not Loading

```tsx
const extraStyles = (window as any).__PD_WIDGET_STYLES__ || "";
```

**Problem:** `__PD_WIDGET_STYLES__` is never set by the Vite build pipeline. The Shadow DOM only receives the hardcoded inline styles from `embed.tsx` (basic reset styles), not the component-specific CSS from `styles.css`. The embedded widget will be **largely unstyled**.

#### No Error Handling on `attachShadow`

```tsx
const shadow = shadowHost.attachShadow({ mode: "open" });
```

**Problem:** `attachShadow` throws if:
- Called twice on the same element
- The element type doesn't support Shadow DOM
- The browser doesn't support Shadow DOM (rare now, but possible in embedded WebViews)

No try/catch wrapper exists.

#### Blunt Cleanup

```tsx
destroy: () => {
    instance.container.innerHTML = "";
    instances.delete(containerId);
}
```

**Problem:** `innerHTML = ""` wipes the entire container contents, including any non-widget content the host page may have placed there. A more precise approach would be to remove only the shadow host element.

---

### 3.3 `styles.css` — Component Styles (200 lines)

#### Positive Aspects

- Uses CSS custom properties for theming (`--primary-color`, `--bg-color`, etc.)
- Responsive design with mobile breakpoints
- Clean BEM-like naming convention

#### Issues

- `--primary-color` is defined but never set from the `primaryColor` prop
- Video mirror transform (`scaleX(-1)`) is CSS-only — canvas capture is not mirrored, which is correct for measurement but confusing for users

---

### 3.4 Configuration Files

#### `vite.config.ts`

```typescript
server: {
    port: 3000,
    proxy: { "/api": "http://localhost:8000" },
    allowedHosts: [".trycloudflare.com"],
}
```

- Correct proxy setup for local development
- `allowedHosts` permits Cloudflare tunnel domains — useful for mobile testing but should be dev-only

#### `vite.widget.config.ts`

- IIFE format for single-file distribution
- `viteStaticCopy` copies MediaPipe WASM files to `dist/widget/mediapipe/`
- Missing: the `locateFile` callback in `PDMeasurer` hardcodes `/mediapipe/face_mesh/${file}` — this path is wrong when the widget is embedded on a third-party site

#### `tsconfig.json`

- `strict: true` — good
- `noUnusedLocals: true` — good
- But TypeScript strictness is undermined by pervasive `as any` casts for MediaPipe globals

---

## 4. Dependency Analysis

### Backend (`requirements.txt`)

| Package | Version | Status |
|---|---|---|
| fastapi | 0.109.0 | Current (minor behind) |
| uvicorn | 0.27.0 | Current |
| python-multipart | 0.0.6 | Current |
| opencv-python-headless | 4.9.0.80 | Current |
| mediapipe | 0.10.9 | Current |
| numpy | 1.26.4 | Pinned — `np.int_` deprecation present |
| Pillow | 10.2.0 | Current |
| pydantic | (via FastAPI) | v2 |

No dev dependencies (no pytest, no linting tools).

### Frontend (`package.json`)

| Package | Version | Status |
|---|---|---|
| react | 18.2.0 | Stable (React 19 available) |
| react-dom | 18.2.0 | Stable |
| typescript | 5.2.2 | Behind (5.7+ available) |
| vite | 5.1.0 | Current major |
| @vitejs/plugin-react | 4.2.1 | Current |
| vite-plugin-static-copy | 1.0.1 | Current |

No test runner (no vitest, no jest). No linting (no eslint). No formatting (no prettier).

---

## 5. Code Quality Summary

### Severity Distribution

| Severity | Count | Category |
|---|---|---|
| **Critical** | 3 | Double yaw correction, broken upload mode, unstyled widget |
| **High** | 5 | Monocular PD 50/50, median off-by-one, CORS misconfig, confidence not pose-corrected, solvePnP unhandled |
| **Medium** | 5 | Fragile URL construction, memory leak, misleading field names, CDN null-safety, coin detection params |
| **Low** | 4 | Deprecated lifecycle, deprecated numpy call, unused prop, duplicate health endpoint |

### File-Level Quality

| File | Quality | Key Concern |
|---|---|---|
| `main.py` | Fair | Median math wrong, monocular override, CORS |
| `face_detection.py` | Good | Solid ML code; fallback path and naming issues |
| `pd_calculator.py` | Poor | Double correction is mathematically wrong |
| `reference_detection.py` | Fair | Inflated confidence, no size-aware detection |
| `PDMeasurer/index.tsx` | Fair | Upload broken, memory leak, stale closures |
| `embed.tsx` | Poor | Styles don't load, no error handling |
| `styles.css` | Good | Clean, responsive, well-organized |
| `types/index.ts` | Good | Proper TypeScript definitions |
| `schemas.py` | Good | Clean Pydantic v2 models |
