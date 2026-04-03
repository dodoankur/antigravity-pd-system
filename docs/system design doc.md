# System Design Document

## Algorithm Analysis: Iris Detection
The system uses the **MediaPipe Iris** model, which is built on top of the 478 landmarks from Face Mesh.
- **Iris Landmarking:** Provides 5 points for each eye (1 center, 4 on the boundary).
- **Scaling Math:**
  1. Calculate `d_pixel` = Euclidean distance between iris center points.
  2. Calculate `iris_w_pixel` = Average width of left and right iris in pixels.
  3. `PD_mm = (d_pixel / iris_w_pixel) * 11.7`.

## API Schemas
### POST /v1/measure
**Request:**
```json
{
  "image_data": "base64...",
  "metadata": {
    "device": "iPhone 13",
    "lighting_score": 0.8
  }
}
```
**Response:**
```json
{
  "pd_binocular": 63.5,
  "pd_monocular": {
    "left": 31.5,
    "right": 32.0
  },
  "confidence_score": 0.98
}
```

## Infrastructure
- **API Gateway:** Nginx or AWS API Gateway.
- **Compute:** Auto-scaling groups of GPU-enabled instances (e.g., AWS g4dn.xlarge) or high-performance CPU instances.
- **Load Balancing:** Round-robin based on instance health.
