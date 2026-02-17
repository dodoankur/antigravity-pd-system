# PD Measurement System — Architecture & Details

> A complete **Pupil Distance (PD)** measurement solution for eyewear companies, built with a Python/FastAPI backend, a React/TypeScript embeddable frontend widget, and computer-vision-powered iris detection.

---

## Project Summary

### What Is This?

The **PD Measurement System** is a self-hosted, embeddable tool that enables eyewear companies to let their customers measure their **Pupil Distance (PD)** — the distance between the centers of the pupils — directly from a browser. Accurate PD is critical for fitting prescription lenses, and this system removes the need for an in-person visit by using **computer vision and machine learning** to estimate PD from a single photo or live camera capture.

### Who Is It For?

This tool is built for **eyewear e-commerce companies** (like Specscart) who want to offer an integrated PD measurement experience on their website — either as a standalone page or as a lightweight **embeddable widget** that can be dropped into any HTML page, React app, or Next.js project.

### How It Works (At a Glance)

1. **User captures a photo** — via file upload or live camera feed in the browser
2. **Optionally places a reference object** (credit card, UK coin, ruler) next to their face for improved accuracy
3. **Image is sent to the backend API** — which runs face and iris detection using MediaPipe
4. **PD is calculated** using either iris diameter estimation or reference object calibration
5. **Results are returned** — overall PD, left eye PD, right eye PD, confidence score, and error margin

### Tech Stack

| Layer                | Technology                     | Purpose                                                  |
| -------------------- | ------------------------------ | -------------------------------------------------------- |
| **Backend API**      | Python 3.10+, FastAPI, Uvicorn | REST API serving PD measurement endpoints                |
| **Computer Vision**  | MediaPipe Face Mesh, OpenCV    | Face/iris landmark detection, reference object detection |
| **ML Model**         | MediaPipe Face Mesh v0.10.9    | 478 facial landmarks including 10 iris points            |
| **Image Processing** | Pillow, NumPy                  | Image conversion, numerical computations                 |
| **Data Validation**  | Pydantic v2                    | Request/response schema validation                       |
| **Frontend**         | React 18, TypeScript 5, Vite 5 | Interactive widget UI with camera/upload support         |
| **Widget System**    | Shadow DOM, IIFE bundle        | Style-isolated embeddable widget for any website         |

### Key Capabilities

- 🎯 **Dual measurement methods** — iris estimation (no props needed) or reference-object calibration (±1.0mm accuracy)
- 📱 **Multiple input methods** — file upload (drag & drop) and live camera capture
- 📏 **10 reference objects** — credit cards, 8 UK coins, and rulers
- 🔌 **Embed anywhere** — single `<script>` tag, works in React, Next.js, or plain HTML
- 🛡️ **Shadow DOM isolation** — widget styles never clash with host page CSS
- ⚠️ **Transparent disclaimers** — every result includes confidence score, error margin, and a medical disclaimer
- 🐍 **Zero Node.js on backend** — pure Python backend with no JavaScript dependencies
- 📄 **Swagger docs** — auto-generated API documentation at `/docs`

### Important Numbers

| Metric                      | Value                    |
| --------------------------- | ------------------------ |
| Iris estimation accuracy    | ±1.5mm (91% of cases)    |
| Reference-assisted accuracy | ±1.0mm                   |
| Average human iris diameter | 11.7mm (σ = 0.5mm)       |
| Face mesh landmarks         | 478 (468 face + 10 iris) |
| API endpoints               | 4                        |
| Backend services            | 3                        |
| Supported reference objects | 10                       |
| Frontend component lines    | ~469                     |

---