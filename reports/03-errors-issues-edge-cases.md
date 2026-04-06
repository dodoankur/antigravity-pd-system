# Errors, Issues & Edge Cases Report

**Project:** Antigravity PD Measurement System
**Date:** 2026-04-04

---

## 1. Critical Bugs

These are bugs that **will produce incorrect results** or **break core functionality**.

### 1.1 Double Yaw Perspective Correction

**Location:** `backend/services/pd_calculator.py:38-63`

**Description:** The yaw-angle perspective correction is applied twice — once to the reference dimension (iris diameter) and once to the measurement (PD distance). This results in an overcorrection of `1/cos²(yaw)` instead of the correct `1/cos(yaw)`.

**Code:**
```python
# Correction #1: Applied to iris diameter
corrected_left_dia  = iris_data.left_iris_diameter_px / np.cos(yaw_rad)
corrected_right_dia = iris_data.right_iris_diameter_px / np.cos(yaw_rad)

# mm_per_pixel is derived from the already-corrected diameter
avg_iris_diameter_px = np.mean([corrected_left_dia, corrected_right_dia])
mm_per_pixel = 11.8 / avg_iris_diameter_px

# Correction #2: Applied AGAIN to the PD measurement
corrected_pd_mm = (pd_pixels / np.cos(yaw_rad)) * mm_per_pixel
```

**Impact:**

| Head Yaw Angle | Correct Correction | Actual Correction | Measurement Error |
|---|---|---|---|
| 0° | 1.000x | 1.000x | 0% |
| 5° | 1.004x | 1.008x | +0.4% |
| 10° | 1.015x | 1.031x | +1.5% |
| 15° | 1.035x | 1.072x | +3.5% |
| 20° | 1.064x | 1.133x | +6.4% |
| 25° | 1.103x | 1.217x | +10.3% |
| 30° | 1.155x | 1.333x | +15.5% |
| 35° (max allowed) | 1.221x | 1.490x | +22.0% |

For a patient with a true PD of 63mm at 35° yaw, the system reports **~77mm** — a 14mm error that would make prescription glasses unusable.

---

### 1.2 Monocular PD Always Returns 50/50 Split

**Location:** `backend/services/pd_calculator.py:67-68` and `backend/main.py:116-117`

**Description:** Both the single-image calculator and the batch endpoint overwrite independently calculable left/right PD values with `overall_pd / 2`.

**Code:**
```python
# pd_calculator.py
left_pd_mm = corrected_pd_mm / 2
right_pd_mm = corrected_pd_mm / 2

# main.py (batch endpoint)
final_result.left_pd_mm = median_pd / 2
final_result.right_pd_mm = median_pd / 2
```

**Impact:** Every user receives perfectly symmetrical monocular PD values. In reality, 85%+ of people have asymmetric monocular PDs (0.5–2mm difference). For progressive lenses or high-index prescriptions, this asymmetry matters clinically.

---

### 1.3 Batch Median Calculation Off-by-One

**Location:** `backend/main.py:111`

**Description:** For even-length arrays, the code takes the upper-middle value instead of the true median.

**Code:**
```python
pd_values = sorted([r.overall_pd_mm for r in results])
median_pd = pd_values[len(pd_values) // 2]  # For 10 values: index 5 (6th element)
```

**Expected:** `(pd_values[4] + pd_values[5]) / 2` for 10 elements.

**Impact:** Slight upward bias in reported PD (typically 0.1–0.5mm). Minor compared to other bugs but still incorrect.

---

### 1.4 Upload Mode Sends Empty Data

**Location:** `frontend/src/components/PDMeasurer/index.tsx:148`

**Description:** When a user uploads an image (instead of using the camera), `handleMultiFrameSubmit` iterates over `captureBufferRef.current` which is never populated in upload mode.

**Code:**
```tsx
// Upload handler sets imageFile but never populates captureBufferRef
const handleFileChange = (e) => {
    setImageFile(e.target.files[0]);
    setImagePreview(URL.createObjectURL(e.target.files[0]));
};

// Submit handler reads from captureBufferRef (always empty for uploads)
const handleMultiFrameSubmit = async () => {
    captureBufferRef.current.forEach((blob, index) => {
        formData.append("images", blob, `frame_${index}.jpg`);
    });
    // Sends FormData with zero images → backend returns error
};
```

**Impact:** The entire upload feature is broken. Users who cannot use their camera (desktop without webcam, permission denied) have no working alternative.

---

### 1.5 Widget Styles Not Loading in Shadow DOM

**Location:** `frontend/src/embed.tsx:78`

**Description:** The Shadow DOM style injection relies on `window.__PD_WIDGET_STYLES__`, a global that is never set by the Vite build pipeline.

**Code:**
```tsx
const extraStyles = (window as any).__PD_WIDGET_STYLES__ || "";  // Always ""
```

**Impact:** The embeddable widget renders with only basic reset styles. All component-specific CSS (layout, colors, typography, responsive breakpoints) is missing. The widget is essentially unusable when embedded on third-party sites.

---

### 1.6 `decomposeProjectionMatrix` Fallback Returns Wrong Data

**Location:** `backend/services/face_detection.py:113`

**Description:** The fallback path for `decomposeProjectionMatrix` returns the rotation matrix (index 1) instead of euler angles (index 6).

**Code:**
```python
decomp = cv2.decomposeProjectionMatrix(proj_matrix)
if len(decomp) >= 7:
    euler_angles = decomp[6]    # Correct
else:
    euler_angles = decomp[1]    # WRONG: rotation matrix (3x3), not euler angles (3x1)
```

**Impact:** If triggered (unlikely on modern OpenCV but possible in edge cases), the entire pose validation chain receives garbage data. Heavily tilted/rotated faces would pass validation and produce wildly incorrect PD measurements.

---

## 2. Security Vulnerabilities

### 2.1 CRITICAL — Exposed API Token

**Location:** `.env.hf` (on disk), `hf_push.sh` (embedded in git remote URL)

**Description:** The Hugging Face API token (`HF_TOKEN=hf_pOx...`) is stored in `.env.hf` and embedded directly into git remote URLs in `hf_push.sh`.

**Risk:**
- If `.env.hf` was ever committed to the remote repo, the token is in git history permanently
- `hf_push.sh` constructs `https://user:TOKEN@huggingface.co/...` — this URL appears in shell history and potentially in `git log`

**Remediation:** Rotate the token immediately. Use `git credential store` or `HF_TOKEN` environment variable instead of embedding in URLs.

---

### 2.2 HIGH — No Upload Size Limit

**Location:** `backend/main.py` — all image upload endpoints

**Description:** `UploadFile` parameters have no `max_size` constraint. A malicious client can upload arbitrarily large files.

**Risk:** Memory exhaustion / denial of service. A single 500MB upload would consume significant RAM during OpenCV/MediaPipe processing.

**Remediation:**
```python
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
contents = await image.read()
if len(contents) > MAX_IMAGE_SIZE:
    raise HTTPException(413, "Image exceeds 10MB limit")
```

---

### 2.3 HIGH — No Rate Limiting

**Location:** All endpoints in `backend/main.py`

**Description:** No rate limiting mechanism exists. The `/api/pd/measure-batch` endpoint runs MediaPipe + OpenCV on up to 10 images per request — highly CPU-intensive.

**Risk:** Trivial compute-based DoS. A simple loop sending requests can saturate the server.

**Remediation:** Add `slowapi` or similar rate limiter (e.g., 10 requests/minute per IP).

---

### 2.4 HIGH — CORS Misconfiguration

**Location:** `backend/main.py:40`

**Description:** `allow_origins=["*"]` combined with `allow_credentials=True` violates the CORS specification. Browsers will reject credentialed requests to wildcard origins.

**Risk:**
- Current: Credentialed cross-origin requests silently fail
- Future: If auth is added, CORS blocks all authenticated frontend calls

**Remediation:** Replace `"*"` with explicit allowed origins.

---

### 2.5 MEDIUM — Content-Type Spoofing

**Location:** `backend/main.py` — image validation

**Description:** The only validation is `content_type.startswith("image/")`. This allows:
- `image/svg+xml` — SVG files (potential XSS vector if stored/served)
- Spoofed content-type headers (file is actually a ZIP/EXE with `image/png` content-type)

**Remediation:** Validate magic bytes (first 4–8 bytes of file) against allowed formats (JPEG, PNG).

---

### 2.6 MEDIUM — Token in Git Remote URL

**Location:** `hf_push.sh:35`

```bash
HF_REMOTE="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE_NAME}"
```

**Risk:** Token appears in `git remote -v` output, shell history, process listing.

---

## 3. Edge Cases & Failure Modes

### 3.1 Client-Side Edge Cases

| # | Scenario | Current Behavior | Expected Behavior |
|---|---|---|---|
| 1 | Google CDN unavailable (outage, firewall, China) | `window.FaceMesh` is undefined → `TypeError` on first camera access, no user feedback | Show error: "Failed to load face detection. Check your connection." |
| 2 | Component unmounts during 10-frame capture | Capture loop continues writing to unmounted refs | Cancel capture loop, clean up resources |
| 3 | User selects file multiple times | Each selection leaks an `ObjectURL` (never revoked) | Revoke previous URL before creating new one |
| 4 | `handleReset` called while camera is tearing down | `initializationLock` blocks re-init → camera never restarts | Queue re-init after teardown completes |
| 5 | `apiEndpoint` contains "/api/pd/measure" as substring in an unexpected position | URL constructed incorrectly (e.g., `/api/pd/measure-batch-v2`) | Use URL parsing, not string replacement |
| 6 | `attachShadow` called twice on same element | Throws `DOMException`, widget never mounts | Catch error, return meaningful error message |
| 7 | Widget embedded on HTTP page, API is HTTPS | Mixed content block — all API calls silently fail | Detect and warn about mixed content |
| 8 | Browser denies camera permission | Camera fails to start | Show clear guidance on enabling camera access |
| 9 | Device has no rear camera (desktop) | Camera may default to screen capture or fail | Handle gracefully with upload fallback |
| 10 | Very slow network (3G) | 10-frame batch upload timeout or extreme latency | Add timeout handling, progress indicator |

### 3.2 Server-Side Edge Cases

| # | Scenario | Current Behavior | Expected Behavior |
|---|---|---|---|
| 1 | Image with no face detected | Returns error (handled) | OK |
| 2 | Image with multiple faces | MediaPipe returns first face | Detect and warn about multiple faces |
| 3 | Very large image (50MB+) | Full image loaded into memory, processed | Reject with 413 before processing |
| 4 | SVG file with `image/svg+xml` content-type | Passes content-type check, OpenCV fails with cryptic error | Reject non-raster formats |
| 5 | Corrupted/truncated JPEG | OpenCV `imdecode` returns `None` → crash on next operation | Check `imdecode` result before proceeding |
| 6 | `solvePnP` fails on degenerate landmarks | Exception propagates, 500 error | Catch, return structured error |
| 7 | Head yaw exactly 90° (profile view) | `cos(90°) = 0` → division by zero in PD calculator | Guard against extreme angles in calculator |
| 8 | Batch with all 10 frames failing validation | Empty `results` list → `sorted()` returns `[]` → `IndexError` on median | Handle empty results gracefully |
| 9 | Concurrent requests exceeding memory | OOM on container | Add request queue or concurrency limit |
| 10 | `HoughCircles` on images with background patterns | False positive circle/line detection (blinds, stripes, etc.) | Add contextual filtering (proximity to face, expected size range) |

### 3.3 User Experience Edge Cases

| # | Scenario | Impact |
|---|---|---|
| 1 | User with strong facial asymmetry | Gets incorrect symmetric 50/50 monocular PD |
| 2 | User wearing thick-frame glasses | Frames may occlude iris landmarks, degrading accuracy |
| 3 | User in low-light environment | Iris landmarks less precise, no low-light warning shown |
| 4 | User with heterochromia or eye conditions | May affect iris detection confidence; no handling |
| 5 | Camera video is mirrored but capture is not | Confusing visual experience (text appears reversed in video but correct in capture) |
| 6 | Error margin exists in API response but never displayed | Users have no indication of measurement quality/reliability |
| 7 | Reference object feature documented but no UI selector | Users cannot access a feature that could improve accuracy |

---

## 4. Validation Gap Analysis

### Frontend vs. Backend Thresholds

| Validation | Frontend Threshold | Backend Threshold | Gap |
|---|---|---|---|
| Horizontal symmetry | 0.45 – 0.55 | 0.35 – 0.65 | Backend 3x more permissive |
| Vertical symmetry | 0.40 – 0.60 | 0.30 – 0.70 | Backend 2x more permissive |
| Yaw angle | (varies) | ±35° | Frontend may differ |
| Pitch angle | (varies) | ±25° | Frontend may differ |

**Risk:** If a client bypasses the frontend (direct API call, custom integration), the backend's looser thresholds accept images the frontend would reject. This means API integrations without the frontend get lower-quality validation.

### Missing Validations

| Check | Frontend | Backend |
|---|---|---|
| File size limit | None | None |
| Magic-byte verification | None | None |
| Image dimension limits | None | None |
| Rate limiting | None | None |
| Multiple face detection | None | None |
| Glasses/occlusion detection | None | None |
| Lighting quality check | None | None |

---

## 5. Error Handling Audit

### Backend Error Handling

| Location | Error Scenario | Handling | Status |
|---|---|---|---|
| `main.py` | Invalid content type | `HTTPException(400)` | OK |
| `main.py` | Image decode failure | `HTTPException(400)` | OK |
| `main.py` | No face detected | Returns error in result | OK |
| `main.py` | All batch frames fail | **No handling** — `IndexError` | MISSING |
| `face_detection.py` | `solvePnP` failure | **No handling** — exception propagates | MISSING |
| `face_detection.py` | `decomposeProjectionMatrix` fallback | Wrong data, no error raised | BUG |
| `pd_calculator.py` | Division by zero (cos=0) | **No handling** — returns `inf` | MISSING |
| `reference_detection.py` | No circles/contours found | Returns `None` | OK |

### Frontend Error Handling

| Location | Error Scenario | Handling | Status |
|---|---|---|---|
| `PDMeasurer/index.tsx` | MediaPipe CDN fails | **No handling** — `TypeError` crash | MISSING |
| `PDMeasurer/index.tsx` | Camera permission denied | Partial — error state set but message unclear | PARTIAL |
| `PDMeasurer/index.tsx` | API request fails | Try/catch with error state | OK |
| `PDMeasurer/index.tsx` | Component unmount during capture | **No handling** — continues executing | MISSING |
| `embed.tsx` | `attachShadow` fails | **No handling** — throws to host page | MISSING |
| `embed.tsx` | React render error | **No error boundary** | MISSING |

---

## 6. Testing Gap

| Category | Expected | Actual |
|---|---|---|
| Backend unit tests | `pytest` for each service | **None** |
| Backend integration tests | API endpoint tests | **None** |
| Frontend unit tests | Vitest/Jest for component logic | **None** |
| Frontend integration tests | Component render + interaction | **None** |
| E2E tests | Playwright/Cypress full flow | **None** |
| CI/CD pipeline | Automated on PR | **None** |
| Accuracy tests | Known-input PD validation | **None** |
| Load tests | Concurrent request handling | **None** |

The `docs/engineering/12-testing-strategy.md` describes a comprehensive testing plan, but **zero tests exist** in the codebase.
