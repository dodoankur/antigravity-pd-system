# MVP Technical Documentation

## Tech Stack
- **Frontend:** React with TypeScript.
- **CV Framework:** MediaPipe (Face Mesh + Iris).
- **Backend:** Python (FastAPI).
- **Infrastructure:** Docker, AWS/GCP.
- **Database:** PostgreSQL (for storing anonymized results/metrics).

## Core Features
1. **Real-time Face Detection:** Guided UI to ensure the user is at the correct distance and angle.
2. **Iris Tracking:** Sub-pixel detection of pupil centers and iris boundaries.
3. **Card-less Calibration:** Using the HVID constant for millimeter conversion.
4. **Instant Results:** Results provided within < 5 seconds of capture.

## Limitations
- **Lighting Conditions:** Requires even, bright lighting for accurate iris boundary detection.
- **Device Hardware:** High-quality front-facing camera (720p minimum) is required.
- **Iris Variance:** While HVID is relatively constant (11.7mm ± 0.5mm), extreme outliers can introduce a 2-4% error.

## Accuracy Benchmarks
- **Target Accuracy:** ±1.0mm (standard optical grade).
- **Validation:** Benchmarked against manual pupilometer measurements.
