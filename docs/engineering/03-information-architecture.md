# 03. Information Architecture

## User Flow
1. **Landing Page:** Introduction and "Start Measurement" CTA.
2. **Permissions:** Browser request for Camera access.
3. **Tutorial:** Static or animated guide on posture and lighting.
4. **Active Scanning:** Real-time feedback loop (MediaPipe running locally).
5. **Capture:** Automatic or manual capture once alignment is optimal.
6. **Processing:** Loading state while server-side scoring engine runs.
7. **Result:** Display PD (Binocular and Monocular) with "Retake" option.

## Data Model (Client-side)
- `SessionState`: { id, status: 'idle'|'scanning'|'processing'|'completed', error: string | null }
- `MeasurementFrame`: { timestamp, landmarks: [], lightingScore: number }
