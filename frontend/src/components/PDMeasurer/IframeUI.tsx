import React, { useCallback, useEffect, useRef, useState } from "react";
import { PDMeasurerProps, PDMeasurementResult } from "../../types";
import "./iframe-ui.css";

// ─── Icons ────────────────────────────────────────────────────────────────────

const EyeIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
        <circle cx="12" cy="12" r="3" />
    </svg>
);

// ─── Component ────────────────────────────────────────────────────────────────

/**
 * IframeUI — A full-screen, Lenskart-inspired PD measurement UI
 * designed exclusively for the embedded / iframe context inside the POS portal.
 *
 * All measurement logic is delegated to the backend; this component owns only
 * the camera capture, result display, and postMessage handoff.
 */
export const IframeUI: React.FC<PDMeasurerProps> = ({
    apiEndpoint = "/api/pd/measure-batch",
    onMeasurement,
    onError,
    mediapipeBasePath = "/mediapipe/face_mesh",
}) => {
    type Step = "capture" | "processing" | "results" | "error";

    const videoRef  = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const [step,            setStep]            = useState<Step>("capture");
    const [result,          setResult]          = useState<PDMeasurementResult | null>(null);
    const [errorMsg,        setErrorMsg]        = useState<string>("");
    const [instruction,     setInstruction]     = useState<string>("Position your face inside the frame");
    const [isValid,         setIsValid]         = useState<boolean>(false);
    const [isCapturing,     setIsCapturing]     = useState<boolean>(false);
    const [captureProgress, setCaptureProgress] = useState<number>(0);

    // MediaPipe + face-mesh refs
    const faceMeshRef    = useRef<any>(null);
    const cameraRef      = useRef<any>(null);
    const capturedFrames = useRef<string[]>([]);
    const warmupCount    = useRef<number>(0);
    const TOTAL_FRAMES   = 10;
    const WARMUP_FRAMES  = 3;

    // ── Camera ────────────────────────────────────────────────────────────────

    const stopCamera = useCallback(() => {
        cameraRef.current?.stop();
        streamRef.current?.getTracks().forEach(t => t.stop());
        streamRef.current = null;
    }, []);

    const startCamera = useCallback(async () => {
        if (!videoRef.current) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "user" },
            });
            streamRef.current = stream;
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
        } catch {
            setErrorMsg("Camera access denied. Please allow camera permissions and try again.");
            setStep("error");
        }
    }, []);

    // ── MediaPipe face-mesh (validation only, not measurement) ────────────────

    const initFaceMesh = useCallback(() => {
        const win = window as any;
        if (!win.FaceMesh || !win.Camera) return;

        const fm = new win.FaceMesh({
            locateFile: (file: string) => `${mediapipeBasePath}/${file}`,
        });
        fm.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });

        fm.onResults((res: any) => {
            if (warmupCount.current < WARMUP_FRAMES) { warmupCount.current++; return; }

            const detected = res.multiFaceLandmarks && res.multiFaceLandmarks.length > 0;

            if (!detected) {
                setIsValid(false);
                setInstruction("No face detected — look straight at the camera");
                (window as any).__pdFaceW = 0;
                return;
            }

            // ── 1. SIZE CHECK ─────────────────────────────────────────────────
            // faceW is normalised 0..1 relative to frame width
            const lms  = res.multiFaceLandmarks[0];
            const xs   = lms.map((l: any) => l.x);
            const ys   = lms.map((l: any) => l.y);
            const faceW = Math.max(...xs) - Math.min(...xs);
            const actualW = videoRef.current?.videoWidth ?? 1280;

            // Expose for live tuning via console: window.__pdFaceW, window.__pdActualW
            (window as any).__pdFaceW   = faceW;
            (window as any).__pdActualW = actualW;

            // Resolution-scaled thresholds (baseline at 1280px stream width):
            //   Calibrated from real measurements (Jun 2026):
            //     faceW=0.154 → one hand away (too far, reject)
            //     faceW=0.217 → arm's length ~60cm (sweet spot ✅)
            //     faceW=0.397 → very close/nose-to-screen (too close, reject)
            //   minFace=0.18 → accept from ~55–70cm away
            //   maxFace=0.35 → reject if closer than ~35cm
            const scale   = 1280 / actualW;
            const minFace = 0.18 * scale;
            const maxFace = 0.35 * scale;

            if (faceW < minFace) {
                setIsValid(false); setInstruction(`Move closer — about an arm's length away [${faceW.toFixed(3)}]`); return;
            }
            if (faceW > maxFace) {
                setIsValid(false); setInstruction(`Too close — move back a little [${faceW.toFixed(3)}]`); return;
            }

            // ── 2. CENTRE-IN-BOX CHECK ────────────────────────────────────────
            // Face centre (normalised 0..1) must sit inside the guide box bounds.
            // Box bounds measured from DOM (normalised to overlay dimensions):
            //   X: 0.36 → 0.64   Y: 0.135 → 0.755
            // Note: MediaPipe x is mirrored (video scaleX(-1)) — flip x for display.
            const rawFaceCx = (Math.min(...xs) + Math.max(...xs)) / 2;
            const rawFaceCy = (Math.min(...ys) + Math.max(...ys)) / 2;
            // Mirror x to match the CSS scaleX(-1) on the video element
            const faceCx = 1 - rawFaceCx;
            const faceCy = rawFaceCy;

            // Guide box normalised bounds (with a small inward tolerance of 0.04)
            const tolerance = 0.04;
            const BOX_X_MIN = 0.36 + tolerance;   // 0.40
            const BOX_X_MAX = 0.64 - tolerance;   // 0.60
            const BOX_Y_MIN = 0.135 + tolerance;  // 0.175
            const BOX_Y_MAX = 0.755 - tolerance;  // 0.715

            if (faceCx < BOX_X_MIN) {
                setIsValid(false); setInstruction("Move your face to the right"); return;
            }
            if (faceCx > BOX_X_MAX) {
                setIsValid(false); setInstruction("Move your face to the left"); return;
            }
            if (faceCy < BOX_Y_MIN) {
                setIsValid(false); setInstruction("Move your face down"); return;
            }
            if (faceCy > BOX_Y_MAX) {
                setIsValid(false); setInstruction("Move your face up"); return;
            }

            setIsValid(true);
            if (!isCapturing) setInstruction(`Hold still — looking good! [${faceW.toFixed(3)}]`);
        });

        faceMeshRef.current = fm;

        const cam = new win.Camera(videoRef.current, {
            onFrame: async () => {
                if (videoRef.current && faceMeshRef.current) {
                    await faceMeshRef.current.send({ image: videoRef.current });
                }
            },
            width: 1280, height: 720,
        });
        cam.start();
        cameraRef.current = cam;
    }, [mediapipeBasePath, isCapturing]);

    useEffect(() => {
        startCamera().then(() => {
            // Wait for scripts to be available
            const interval = setInterval(() => {
                if ((window as any).FaceMesh && (window as any).Camera) {
                    clearInterval(interval);
                    initFaceMesh();
                }
            }, 200);
        });
        return () => stopCamera();
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // ── Capture ───────────────────────────────────────────────────────────────

    const captureFrames = useCallback(async () => {
        if (!videoRef.current || !canvasRef.current) return;
        setIsCapturing(true);
        setInstruction("Hold still — capturing…");
        capturedFrames.current = [];
        setCaptureProgress(0);

        const video  = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width  = video.videoWidth  || 1280;
        canvas.height = video.videoHeight || 720;
        const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

        for (let i = 0; i < TOTAL_FRAMES; i++) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            capturedFrames.current.push(canvas.toDataURL("image/png"));
            setCaptureProgress(i + 1);
            await new Promise(r => setTimeout(r, 120));
        }

        setIsCapturing(false);
        submitFrames(capturedFrames.current);
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // ── Submit ────────────────────────────────────────────────────────────────

    const submitFrames = useCallback(async (frames: string[]) => {
        setStep("processing");
        stopCamera();

        try {
            const blobs = await Promise.all(frames.map(async (dataUrl) => {
                const res = await fetch(dataUrl);
                return res.blob();
            }));

            const form = new FormData();
            blobs.forEach((blob, i) => form.append("images", blob, `frame_${i}.png`));
            form.append("age_group", "auto");

            const resp = await fetch(apiEndpoint, { method: "POST", body: form });
            if (!resp.ok) throw new Error(`Server error ${resp.status}`);

            const data: PDMeasurementResult = await resp.json();
            setResult(data);
            setStep("results");
            onMeasurement?.(data);
        } catch (err: any) {
            const msg = err?.message ?? "Measurement failed. Please try again.";
            setErrorMsg(msg);
            setStep("error");
            onError?.(msg);
        }
    }, [apiEndpoint, onMeasurement, onError, stopCamera]);

    // ── Reset ─────────────────────────────────────────────────────────────────

    const handleRestart = useCallback(() => {
        setStep("capture");
        setResult(null);
        setErrorMsg("");
        setCaptureProgress(0);
        setIsCapturing(false);
        setIsValid(false);
        warmupCount.current   = 0;
        capturedFrames.current = [];
        setInstruction("Position your face inside the frame");
        startCamera().then(initFaceMesh);
    }, [startCamera, initFaceMesh]);

    // ── Render ────────────────────────────────────────────────────────────────

    return (
        <div className="ifu">

            {/* ── Close button (always visible — posts PD_CLOSE to parent POS modal) ── */}
            <button
                className="ifu__close-btn"
                aria-label="Close"
                onClick={() => {
                    stopCamera();
                    if (window.parent !== window) {
                        window.parent.postMessage({ type: 'PD_CLOSE' }, '*');
                    }
                }}
            >✕</button>

            {/* ── CAPTURE STEP ── */}
            {(step === "capture" || step === "error") && (
                <div className="ifu__camera-wrap">
                    {/* Live video — full bleed */}
                    <video
                        ref={videoRef}
                        className="ifu__video"
                        playsInline
                        muted
                        autoPlay
                    />

                    {/* Hidden canvas for frame capture */}
                    <canvas ref={canvasRef} style={{ display: "none" }} />

                    {/* Face guide box */}
                    <div className="ifu__overlay">
                        <div className={`ifu__guide-box ${isValid ? "ifu__guide-box--ok" : ""} ${isCapturing ? "ifu__guide-box--capturing" : ""}`} />

                        {/* Instruction pill */}
                        <div className={`ifu__instruction ${isValid ? "ifu__instruction--ok" : ""} ${isCapturing ? "ifu__instruction--capturing" : ""}`}>
                            {isCapturing ? (
                                <>
                                    <span>Capturing… {captureProgress}/{TOTAL_FRAMES}</span>
                                    <div className="ifu__progress">
                                        <div className="ifu__progress-bar" style={{ width: `${(captureProgress / TOTAL_FRAMES) * 100}%` }} />
                                    </div>
                                </>
                            ) : (
                                <span>{errorMsg || instruction}</span>
                            )}
                        </div>

                        {/* LIVE SAMPLING badge */}
                        {isCapturing && <div className="ifu__badge">LIVE SAMPLING</div>}
                    </div>

                    {/* CTA button */}
                    {!isCapturing && (
                        <button
                            className={`ifu__btn ifu__btn--capture ${isValid ? "ifu__btn--ready" : "ifu__btn--waiting"}`}
                            onClick={captureFrames}
                            disabled={!isValid}
                        >
                            <span className="ifu__btn-icon"><EyeIcon /></span>
                            {isValid ? "Measure My PD" : "Waiting for face…"}
                        </button>
                    )}
                </div>
            )}

            {/* ── PROCESSING STEP ── */}
            {step === "processing" && (
                <div className="ifu__processing">
                    <div className="ifu__spinner" />
                    <p className="ifu__processing-label">Measuring pupil distance…</p>
                    <p className="ifu__processing-sub">Analysing {TOTAL_FRAMES} frames</p>
                </div>
            )}

            {/* ── RESULTS STEP ── */}
            {step === "results" && result && (
                <div className="ifu__results">
                    {/* Blurred camera-last-frame background effect */}
                    <div className="ifu__results-bg" />

                    <div className="ifu__results-card">
                        {/* Icon */}
                        <div className="ifu__results-icon">
                            <EyeIcon />
                        </div>

                        {/* Main value */}
                        <p className="ifu__results-label">Your pupillary distance is:</p>
                        <p className="ifu__results-value">{result.overall_pd_mm} mm</p>

                        {/* L / R breakdown */}
                        <div className="ifu__results-split">
                            <div className="ifu__results-eye">
                                <span className="ifu__results-eye-label">Right</span>
                                <span className="ifu__results-eye-value">{result.right_pd_mm} mm</span>
                            </div>
                            <div className="ifu__results-divider" />
                            <div className="ifu__results-eye">
                                <span className="ifu__results-eye-label">Left</span>
                                <span className="ifu__results-eye-value">{result.left_pd_mm} mm</span>
                            </div>
                        </div>

                        {/* Confidence */}
                        <p className="ifu__results-confidence">
                            Confidence: {(result.confidence_score * 100).toFixed(0)}% &nbsp;·&nbsp; ±{result.error_margin.value_mm.toFixed(1)} mm
                        </p>

                        {/* Asymmetry warning */}
                        {result.asymmetry_warning && (
                            <p className="ifu__results-warning">
                                ⚠️ Larger-than-usual L/R difference. Consider re-measuring.
                            </p>
                        )}
                    </div>

                    {/* Actions */}
                    <div className="ifu__results-actions">
                        <button className="ifu__btn ifu__btn--save" onClick={() => onMeasurement?.(result)}>
                            Save my PD
                        </button>
                        <button className="ifu__btn ifu__btn--restart" onClick={handleRestart}>
                            Restart measurement
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default IframeUI;
