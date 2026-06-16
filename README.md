# PD Measurement System

A complete pupil distance (PD) measurement solution for eyewear companies. Measures overall PD, left eye PD, and right eye PD from facial images using computer vision.

## Features

- 🎯 **Accurate Measurements**: Uses MediaPipe Face Mesh with iris detection + perspective correction
- 📱 **Multiple Input Methods**: File upload and live camera capture (10-frame batch)
- 📏 **Reference Object Support**: Credit cards, GBP coins, and rulers for improved accuracy
- 🔌 **Embeddable Widget**: Works with React, Next.js, and vanilla HTML/JS via Shadow DOM
- ⚠️ **Transparent Disclaimers**: Clear error margins and confidence scores
- 🐍 **Python Backend**: FastAPI with no Node.js dependencies
- 🛡️ **Optional API Key Auth**: Set `API_KEY` env var to protect the endpoint
- 📝 **Structured Logging**: Every request logged with method, status, and latency

---

## Quick Start

### Prerequisites

- Python 3.12 (the `.venv` in `backend/` is already set up for 3.12)
- Node.js 18+ and npm 9+

---

### Backend Setup

```bash
cd backend

# Option A — use the existing .venv (recommended, already has all deps)
.venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Option B — create a fresh venv
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

- Swagger docs:  http://localhost:8000/docs
- Health check:  http://localhost:8000/api/health

#### Environment Variables (optional)

| Variable       | Default                                    | Description                                      |
| -------------- | ------------------------------------------ | ------------------------------------------------ |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated list of allowed frontend origins |
| `API_KEY`      | _(unset — auth disabled)_                  | Set to require `X-API-Key` header on all requests |

```bash
# Example — enable API key and allow production frontend
export CORS_ORIGINS="https://specscart.co.uk,http://localhost:5173"
export API_KEY="your-secret-key"
.venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

### Frontend Setup

```bash
cd frontend

# Install dependencies (skip if node_modules already exists)
npm install

# Run development server (starts on http://localhost:5173)
npm run dev
```

The frontend will be available at http://localhost:5173

> Note: Vite uses port **5173** by default (not 3000). The backend CORS config
> already allows both ports.

---

### Run Both Together (split terminal)

Terminal 1 — Backend:
```bash
cd backend
.venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 — Frontend:
```bash
cd frontend
npm run dev
```

---

### Build Embeddable Widget

```bash
cd frontend
npm run build:widget
```

Output: `frontend/dist/widget/pd-widget.js`

Embed it on any page with a single script tag (see Integration section below).

---

### Run Tests

```bash
cd backend

# Using the .venv Python 3.12 runtime
.venv/bin/python -m pip install pytest -q
.venv/bin/python -m pytest tests/ -v

# Or using uv (if installed)
uv run pytest tests/ -v
```

---

### Lint (ruff)

```bash
cd backend

# Check
.venv/bin/python -m ruff check .

# Auto-fix
.venv/bin/python -m ruff check --fix .

# Format
.venv/bin/python -m ruff format .
```

---

## Integration

### React / Next.js

```tsx
import { PDMeasurer } from "./components/PDMeasurer";
import "./components/PDMeasurer/styles.css";

function App() {
    return (
        <PDMeasurer
            apiEndpoint="https://your-api.com/api/pd/measure"
            onMeasurement={(result) => {
                console.log("PD:", result.overall_pd_mm);
                console.log("Left:", result.left_pd_mm);
                console.log("Right:", result.right_pd_mm);
            }}
            onError={(error) => console.error(error)}
        />
    );
}
```

### Vanilla HTML/JavaScript

```html
<div id="pd-widget"></div>
<!-- MediaPipe must be loaded before the widget bundle -->
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
<script src="https://your-cdn.com/pd-widget.js"></script>
<script>
    PDMeasurementWidget.init("#pd-widget", {
        apiEndpoint: "https://your-api.com/api/pd/measure",
        onMeasurement: function (result) {
            console.log("Overall PD:", result.overall_pd_mm + "mm");
            console.log("Left PD:",    result.left_pd_mm    + "mm");
            console.log("Right PD:",   result.right_pd_mm   + "mm");
        },
        onError: function(err) {
            console.error("PD error:", err);
        }
    });
</script>
```

---

## API Endpoints

### `POST /api/pd/measure`

Measure PD from a single uploaded image.

**Request (multipart/form-data):**

| Field            | Type   | Required | Description                          |
| ---------------- | ------ | -------- | ------------------------------------ |
| `image`          | file   | Yes      | Face image (JPEG, PNG, or WebP)      |
| `reference_type` | string | No       | Reference object type (default: none) |

**Response:**

```json
{
    "overall_pd_mm": 63.5,
    "left_pd_mm": 31.8,
    "right_pd_mm": 31.7,
    "method": "iris_estimation",
    "model_used": "MediaPipe v0.10 + Perspective Correction Engine",
    "confidence_score": 0.88,
    "error_margin": {
        "value_mm": 1.5,
        "percentage": 2.4,
        "confidence_score": 0.88
    },
    "disclaimer": "Calculated using 3D Perspective Correction. Confidence: 88%",
    "face_detected": true,
    "eyes_detected": true,
    "reference_detected": null,
    "reference_scale_factor": null
}
```

### `POST /api/pd/measure-batch`

Measure PD from multiple frames and return the median result for stability.
This is what the frontend camera mode uses internally (10 frames).

**Request:** same as `/api/pd/measure` but `images` (plural, multiple files).

### `GET /api/reference-types`

Returns the list of supported reference objects with labels and accuracy info.

### `GET /api/health`

Returns `{"status": "healthy", "version": "1.0.0"}` when all services are up,
or `{"status": "degraded", ...}` if ML services failed to initialise.

---

## Reference Objects

| Type            | Description                    | Accuracy |
| --------------- | ------------------------------ | -------- |
| `none`          | Iris estimation (no reference) | ±1.5mm   |
| `credit_card`   | Standard ISO credit card       | ±1.0mm   |
| `coin_gbp_1p`   | 1 penny (20.3mm)               | ±1.0mm   |
| `coin_gbp_2p`   | 2 pence (25.9mm)               | ±1.0mm   |
| `coin_gbp_5p`   | 5 pence (18.0mm)               | ±1.0mm   |
| `coin_gbp_10p`  | 10 pence (24.5mm)              | ±1.0mm   |
| `coin_gbp_20p`  | 20 pence (21.4mm)              | ±1.0mm   |
| `coin_gbp_50p`  | 50 pence (27.3mm)              | ±1.0mm   |
| `coin_gbp_1`    | £1 coin (23.43mm)              | ±1.0mm   |
| `coin_gbp_2`    | £2 coin (28.4mm)               | ±1.0mm   |
| `ruler`         | Ruler with 10mm segments       | ±1.0mm   |

---

## Accuracy & Disclaimers

### Face-Only Mode (Iris Estimation)

- Uses average human iris diameter (11.7mm) as scale anchor
- Applies cos(yaw) perspective correction for turned heads
- Accuracy: ±1.5mm in 91% of cases
- Confidence affected by: lighting, image quality, face angle

### Reference-Assisted Mode

- Uses detected reference object for direct scale calibration
- Accuracy: ±1.0mm
- Requires reference object to be in the same focal plane as the face

### Important Notice

> This measurement is intended for general guidance only. For prescription
> eyewear, always verify your PD with a qualified optician or eye care
> professional.

---

## Project Structure

```
antigravity-pd-project/
├── backend/
│   ├── main.py                    # FastAPI app, endpoints, middleware
│   ├── requirements.txt           # pip dependencies
│   ├── pyproject.toml             # project config + ruff lint rules
│   ├── conftest.py                # pytest path setup
│   ├── tests/
│   │   └── test_pd_system.py      # 18 unit tests
│   ├── models/
│   │   └── schemas.py             # Pydantic v2 request/response models
│   └── services/
│       ├── face_detection.py      # MediaPipe Face Mesh + pose estimation
│       ├── reference_detection.py # Reference object detection (card/coin/ruler)
│       └── pd_calculator.py       # PD calculation + perspective correction
├── frontend/
│   ├── package.json
│   ├── vite.config.ts             # Dev server config
│   ├── vite.widget.config.ts      # Widget IIFE bundle config
│   └── src/
│       ├── main.tsx               # Dev app entry point
│       ├── embed.tsx              # Widget entry point (global PDMeasurementWidget)
│       ├── types/index.ts         # Shared TypeScript types
│       └── components/
│           └── PDMeasurer/        # Main widget component (~580 lines)
│               ├── index.tsx
│               └── styles.css
├── examples/
│   ├── vanilla-html/              # Plain HTML integration example
│   └── nextjs/                    # Next.js integration example
├── docs/
├── reports/
├── docker-compose.yml
└── README.md
```

---

## Docker

```bash
# Build and run both services
docker-compose up --build

# Backend only
cd backend
docker build -t pd-backend .
docker run -p 8000:8000 pd-backend

# Frontend only
cd frontend
docker build -t pd-frontend .
docker run -p 5173:5173 pd-frontend
```

---

## Future Enhancements (Phase 2)

- [ ] WebXR Depth Sensing for AR-enabled measurement
- [ ] Multi-face support
- [ ] Measurement history storage
- [ ] Real-time video tracking
- [ ] EU/US coin reference objects

---

## License

Proprietary — For eyewear company use only.
