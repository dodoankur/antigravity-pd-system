# Architecture Analysis Report

**Project:** Antigravity PD Measurement System
**Date:** 2026-04-04

---

## 1. Overview

This is a **monorepo** with two independently deployable services targeting an embeddable PD (Pupillary Distance) measurement widget.

```
antigravity-pd-project/
├── backend/          Python 3.12 / FastAPI / MediaPipe / OpenCV → Hugging Face Spaces (Docker)
├── frontend/         React 18 / TypeScript / Vite → Vercel
├── examples/         Vanilla HTML + Next.js integration demos
└── docs/             12 engineering spec documents
```

---

## 2. System Architecture

### Processing Model

**Hybrid processing** — real-time feedback runs client-side, measurement computation runs server-side:

```
Browser (Client)
 ├── MediaPipe FaceMesh (global CDN script)  ← real-time pose validation (~30fps)
 ├── Canvas frame capture (10 frames @ 100ms intervals)
 └── POST /api/pd/measure-batch (blob array)
         ↓
FastAPI Backend (Server)
 ├── FaceDetectionService   (MediaPipe FaceMesh pass #2)
 ├── PDCalculatorService    (iris-based measurement + perspective correction)
 └── ReferenceDetectionService (card/coin/ruler detection via OpenCV)
         ↓
JSON Response → Browser renders results
```

### Core Flow

1. Browser loads MediaPipe FaceMesh via Google CDN `<script>` tags
2. Real-time 5-layer pose validation runs at ~30fps (symmetry, tilt, distance, stability, face presence)
3. Once a valid pose is held, 10 frames are auto-captured at 100ms intervals
4. Frames are sent as a batch (`POST /api/pd/measure-batch`) to the Python backend
5. Backend runs its own MediaPipe FaceMesh pass + pose validation + PD calculation on each frame
6. Median PD across all valid frames is computed and returned as JSON
7. Results (overall PD, left PD, right PD, confidence) are displayed in the UI

---

## 3. Deployment Architecture

| Layer | Service | Platform | Config |
|---|---|---|---|
| Backend API | FastAPI + Docker | Hugging Face Spaces | Port 7860, Python 3.12-slim |
| Frontend App | React + Vite | Vercel | `VITE_API_BASE_URL` env var |
| Widget Delivery | IIFE bundle + Shadow DOM | Self-hosted / Vercel | `pd-widget.js` + `style.css` |
| Dev Proxy | Vite dev server | Local | Port 3000, proxies `/api/*` → `localhost:8000` |

### Deployment Scripts

- `hf_push.sh` — Bash script that pushes the `backend/` directory as a standalone git repo to Hugging Face Spaces
- Frontend deployed via Vercel's automatic Git integration

---

## 4. Backend Architecture

### Tech Stack

| Component | Technology | Version |
|---|---|---|
| Framework | FastAPI | 0.109.0 |
| Runtime | Python | 3.12 |
| ML / Vision | MediaPipe | 0.10.9 |
| Image Processing | OpenCV (headless) | 4.9.0.80 |
| Validation | Pydantic | v2 (via FastAPI) |
| Math | NumPy | 1.26.4 |
| Server | Uvicorn | 0.27.0 |
| Container | Docker | Python 3.12-slim base |

### Service Layer Design

```
main.py (FastAPI app)
 ├── FaceDetectionService    → services/face_detection.py
 │   ├── MediaPipe FaceMesh (478-point model, refine_landmarks=True)
 │   ├── Head pose estimation (cv2.solvePnP + decomposeProjectionMatrix)
 │   ├── Iris diameter measurement (median of 4 edge distances)
 │   └── Advanced confidence scoring (weighted: iris clarity, symmetry, pose, face size)
 │
 ├── PDCalculatorService     → services/pd_calculator.py
 │   ├── Iris-based PD calculation (11.8mm average iris diameter reference)
 │   ├── Yaw perspective correction (cos correction for head angle)
 │   └── Reference-object-based PD calculation (card/coin/ruler anchor)
 │
 └── ReferenceDetectionService → services/reference_detection.py
     ├── Credit card detection (contour analysis + aspect ratio matching)
     ├── Coin detection (HoughCircles + known diameter lookup)
     └── Ruler detection (vertical line detection + spacing analysis)
```

### API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/api/health` | Health check (duplicate) |
| `POST` | `/api/pd/measure` | Single image PD measurement |
| `POST` | `/api/pd/measure-batch` | Multi-image batch PD measurement (primary) |
| `GET` | `/api/reference-types` | List supported reference object types |

### Data Models (Pydantic v2)

- `FaceDetectionResult` — landmarks, iris data, pose angles, confidence
- `IrisData` — left/right iris center + diameter in pixels
- `PDResult` — overall PD, left PD, right PD, confidence, error margin
- `ReferenceDetectionResult` — detected object type, dimensions, confidence

---

## 5. Frontend Architecture

### Tech Stack

| Component | Technology | Version |
|---|---|---|
| Framework | React | 18.2.0 |
| Language | TypeScript | 5.2.2 |
| Bundler | Vite | 5.1.0 |
| ML (Client) | MediaPipe FaceMesh | CDN (global script) |
| Widget Format | IIFE + Shadow DOM | Custom build config |

### Component Architecture

```
frontend/src/
 ├── main.tsx              → Dev entry point (standalone page)
 ├── embed.tsx             → Widget entry point (Shadow DOM + global API)
 ├── types/index.ts        → TypeScript type definitions
 └── components/
     └── PDMeasurer/
         ├── index.tsx     → Main component (437 lines, 3-step state machine)
         └── styles.css    → Component CSS with CSS custom properties
```

### State Machine

```
┌─────────┐    auto-capture    ┌────────────┐    API response    ┌─────────┐
│ CAPTURE  │ ────────────────► │ PROCESSING │ ──────────────────► │ RESULTS │
│          │  (10 frames)      │            │                     │         │
└─────────┘                    └────────────┘                     └─────────┘
     ▲                                                                 │
     └─────────────────── handleReset() ◄──────────────────────────────┘
```

### Widget Distribution

Two build configurations:
- `vite.config.ts` — Standard dev/production build (ES modules, code splitting)
- `vite.widget.config.ts` — IIFE widget build (single file, Shadow DOM encapsulation)

Widget API:
```javascript
window.PDWidget.init({
  containerId: "pd-container",
  apiEndpoint: "https://your-api.com/api/pd/measure",
  primaryColor: "#007bff",
  onResult: (result) => { /* handle PD result */ }
});
```

---

## 6. Architectural Strengths

| Strength | Details |
|---|---|
| Clean service separation | Backend has distinct services for face detection, PD calculation, and reference detection |
| Shadow DOM isolation | Widget uses Shadow DOM to prevent CSS bleed on third-party sites |
| Batch measurement | 10-frame median approach reduces single-frame noise significantly |
| Pydantic v2 schemas | Proper request/response typing with validation |
| Dual build system | Separate Vite configs for app and widget builds |
| Comprehensive documentation | 12 engineering spec documents covering product, architecture, API, testing |
| Real-time feedback | Client-side pose validation gives users immediate guidance before capture |

---

## 7. Architectural Weaknesses

### 7.1 Redundant MediaPipe Processing

MediaPipe FaceMesh runs **twice** for every measurement — once in the browser for pose validation, and again on the server for PD calculation. This doubles the computational cost. The client-side landmarks could be sent to the server instead of raw images, halving latency and server load.

### 7.2 No Local Development Orchestration

There is no `docker-compose.yml` for running the full stack locally. Developers must manually start backend and frontend in separate terminals with correct environment variables.

### 7.3 Built Artifacts in Source Control

`frontend/dist/` is committed to the repository despite `dist/` being listed in `.gitignore`. This was likely force-added before the gitignore rule existed. Build artifacts should never be in source control — they bloat the repo, create merge conflicts, and mask build issues.

### 7.4 No CI/CD Pipeline

Zero CI/CD configuration exists — no `.github/workflows/`, no `.gitlab-ci.yml`. Deployments are manual (`hf_push.sh` for backend, Vercel auto-deploy for frontend). No automated linting, type-checking, or testing gates.

### 7.5 No Authentication Layer

All API endpoints are entirely public. The `/api/pd/measure-batch` endpoint is computationally expensive (MediaPipe + OpenCV per frame, 10 frames per request). Without authentication or rate limiting, the API is trivially abusable.

### 7.6 CDN-Dependent Frontend

MediaPipe is loaded via Google CDN `<script>` tags in `index.html`. If the CDN is unavailable (outage, corporate firewall, China), the entire application fails silently — no error message, no fallback.

### 7.7 Tightly Coupled URL Construction

The frontend constructs the batch endpoint URL by string-replacing the single-image URL. This creates a fragile coupling — any change to the API path structure breaks the frontend URL construction logic.

### 7.8 Missing docker-compose for Multi-Service Dev

No `docker-compose.yml` means developers must manually coordinate backend and frontend startup, port configuration, and environment variables.

### 7.9 No Error Boundary

The React component has no error boundary. A crash in MediaPipe, canvas operations, or API handling will unmount the entire widget with no user feedback.

---

## 8. Architecture Diagram

```
                    ┌──────────────────────────────┐
                    │         User's Browser        │
                    │                                │
                    │  ┌──────────────────────────┐  │
                    │  │     PDMeasurer Component  │  │
                    │  │                            │  │
                    │  │  ┌────────────────────┐   │  │
                    │  │  │  MediaPipe FaceMesh │   │  │  ◄── Google CDN
                    │  │  │  (Pose Validation)  │   │  │
                    │  │  └────────────────────┘   │  │
                    │  │           │                 │  │
                    │  │  ┌────────────────────┐   │  │
                    │  │  │  Canvas Capture     │   │  │
                    │  │  │  (10 frames/100ms)  │   │  │
                    │  │  └────────────────────┘   │  │
                    │  └──────────┬───────────────┘  │
                    └─────────────┼──────────────────┘
                                  │ POST /api/pd/measure-batch
                                  │ (multipart/form-data)
                                  ▼
                    ┌──────────────────────────────┐
                    │   Hugging Face Spaces (Docker)│
                    │                                │
                    │  ┌──────────────────────────┐  │
                    │  │      FastAPI (main.py)    │  │
                    │  │                            │  │
                    │  │  ┌─────────────────────┐  │  │
                    │  │  │ FaceDetectionService │  │  │
                    │  │  │ (MediaPipe + OpenCV) │  │  │
                    │  │  └─────────────────────┘  │  │
                    │  │  ┌─────────────────────┐  │  │
                    │  │  │ PDCalculatorService  │  │  │
                    │  │  │ (Iris PD + Correct.) │  │  │
                    │  │  └─────────────────────┘  │  │
                    │  │  ┌─────────────────────┐  │  │
                    │  │  │ ReferenceDetection   │  │  │
                    │  │  │ (Card/Coin/Ruler)    │  │  │
                    │  │  └─────────────────────┘  │  │
                    │  └──────────────────────────┘  │
                    └──────────────────────────────┘
```

---

## 9. File-Level Dependency Map

### Backend Dependencies

```
main.py
 ├── fastapi, fastapi.middleware.cors
 ├── models.schemas (Pydantic models)
 ├── services.face_detection.FaceDetectionService
 ├── services.pd_calculator.PDCalculatorService
 └── services.reference_detection.ReferenceDetectionService

face_detection.py
 ├── mediapipe (mp.solutions.face_mesh)
 ├── cv2 (solvePnP, decomposeProjectionMatrix)
 └── numpy

pd_calculator.py
 ├── numpy (cos, radians, mean)
 └── models.schemas (IrisData, PDResult)

reference_detection.py
 ├── cv2 (HoughCircles, findContours, minAreaRect, HoughLinesP)
 └── numpy
```

### Frontend Dependencies

```
PDMeasurer/index.tsx
 ├── react (useState, useEffect, useRef, useCallback)
 ├── window.FaceMesh (global — MediaPipe CDN)
 ├── window.Camera (global — MediaPipe CDN)
 └── ./styles.css

embed.tsx
 ├── react, react-dom/client
 ├── PDMeasurer component
 └── Shadow DOM API

vite.config.ts → dev server + proxy
vite.widget.config.ts → IIFE build + static file copy
```
