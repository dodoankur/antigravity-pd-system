# 06. API Contracts

## Endpoint: `POST /v1/score`
**Description:** Processes a high-resolution frame and returns precise PD.

**Request Body:**
```json
{
  "image": "string (base64)",
  "landmarks_hint": {
    "left_eye": [x, y],
    "right_eye": [x, y]
  },
  "session_id": "string"
}
```

**Response (200 OK):**
```json
{
  "pd": 64.2,
  "monocular_pd": { "l": 32.1, "r": 32.1 },
  "confidence": 0.99,
  "timestamp": "2026-02-17T22:00:00Z"
}
```
