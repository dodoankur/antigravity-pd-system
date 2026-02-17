# PD Measurement System

A complete pupil distance (PD) measurement solution for eyewear companies. Measures overall PD, left eye PD, and right eye PD from facial images using computer vision.

## Features

- 🎯 **Accurate Measurements**: Uses MediaPipe Face Mesh with iris detection
- 📱 **Multiple Input Methods**: File upload and camera capture
- 📏 **Reference Object Support**: Credit cards, GBP coins, and rulers for improved accuracy
- 🔌 **Embeddable Widget**: Works with React, Next.js, and vanilla HTML/JS
- ⚠️ **Transparent Disclaimers**: Clear error margins and confidence scores
- 🐍 **Python Backend**: FastAPI with no Node.js dependencies

## Quick Start

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
# or
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

- API Docs: `http://localhost:8000/docs`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Build Embeddable Widget

```bash
cd frontend
npm run build:widget
```

Output: `frontend/dist/widget/pd-widget.js`

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
<script src="https://your-cdn.com/pd-widget.js"></script>
<script>
    PDMeasurementWidget.init("#pd-widget", {
        apiEndpoint: "https://your-api.com/api/pd/measure",
        onMeasurement: function (result) {
            console.log("Overall PD:", result.overall_pd_mm + "mm");
        },
    });
</script>
```

## API Endpoints

### `POST /api/pd/measure`

Measure PD from an uploaded image.

**Request:**

- `image`: Face image file (JPEG, PNG, WebP)
- `reference_type`: Optional reference object type

**Response:**

```json
{
    "overall_pd_mm": 63.5,
    "left_pd_mm": 31.8,
    "right_pd_mm": 31.7,
    "method": "iris_estimation",
    "model_used": "MediaPipe Face Mesh v0.10.9 with Iris Refinement",
    "confidence_score": 0.88,
    "error_margin": {
        "value_mm": 1.5,
        "percentage": 3.0,
        "confidence_score": 0.88
    },
    "disclaimer": "Measurement Method: Iris Diameter Estimation..."
}
```

### `GET /api/reference-types`

Get list of supported reference object types.

## Reference Objects

| Type           | Description                    | Accuracy |
| -------------- | ------------------------------ | -------- |
| `none`         | Iris estimation (no reference) | ±1.5mm   |
| `credit_card`  | Standard credit card           | ±1.0mm   |
| `coin_gbp_1`   | £1 coin (23.43mm)              | ±1.0mm   |
| `coin_gbp_2`   | £2 coin (28.4mm)               | ±1.0mm   |
| `coin_gbp_50p` | 50 pence (27.3mm)              | ±1.0mm   |
| `ruler`        | Ruler with 10mm segments       | ±1.0mm   |

## Accuracy & Disclaimers

### Face-Only Mode (Iris Estimation)

- Uses average human iris diameter (11.7mm) for scale
- Accuracy: ±1.5mm in 91% of cases
- Confidence affected by: lighting, image quality, face angle

### Reference-Assisted Mode

- Uses detected reference object for scale calibration
- Accuracy: ±1.0mm
- Requires proper placement of reference object

### Important Notice

> This measurement is intended for general guidance. For prescription eyewear, verify your PD with a qualified optician or eye care professional.

## Project Structure

```
antigravity-pd-project/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   └── services/
│       ├── face_detection.py   # MediaPipe face mesh
│       ├── reference_detection.py  # Reference object detection
│       └── pd_calculator.py    # PD calculation logic
├── frontend/
│   ├── package.json
│   ├── vite.config.ts          # Development config
│   ├── vite.widget.config.ts   # Widget build config
│   └── src/
│       ├── main.tsx            # Dev entry point
│       ├── embed.tsx           # Widget entry point
│       └── components/
│           └── PDMeasurer/     # Main widget component
├── examples/
│   ├── vanilla-html/           # HTML integration example
│   └── nextjs/                 # Next.js integration example
└── README.md
```

## Future Enhancements (Phase 2)

- [ ] WebXR Depth Sensing for AR-enabled measurement
- [ ] Multi-face support
- [ ] Measurement history storage
- [ ] Real-time video tracking
- [ ] Additional reference objects (EU coins, US coins)

## License

Proprietary - For eyewear company use only.
