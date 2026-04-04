---
title: PD Measurement API
emoji: 📏
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# PD Measurement API Backend

This is the high-performance backend for the Pupil Distance (PD) Measurement tool, hosted on Hugging Face Spaces using Docker.

## Tech Stack
-   **FastAPI**: High-performance web framework.
-   **OpenCV**: Image processing.
-   **MediaPipe**: Face mesh and landmark detection.

## API Endpoint
The measurement endpoint is at `/api/pd/measure` (POST).
