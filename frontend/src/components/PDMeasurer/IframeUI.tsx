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

const UploadIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <polyline points="16 16 12 12 8 16" />
        <line x1="12" y1="12" x2="12" y2="21" />
        <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
    </svg>
);

const FlipCameraIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 7h-3a2 2 0 0 1-2-2V2" />
        <path d="M9 2H4a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9l-5-5z" />
        <circle cx="12" cy="14" r="3" />
        <polyline points="7 10 7 7 10 7" />
    </svg>
);

// ─── Component ────────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export const IframeUI: React.FC<PDMeasurerProps> = ({
    apiEndpoint = `${API_BASE}/api/pd/measure-batch`,
    onMeasurement,
    onError,
    mediapipeBasePath = "/mediapipe/face_mesh",
}) => {
    type Step = "capture" | "upload" | "processing" | "results" | "error";

    const videoRef  = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const [step,            setStep]            = useState<Step>("capture");
    const [result,          setResult]          = useState<PDMeasurementResult | null>(null);
    const [errorMsg,        setErrorMsg]        = useState<string>("");
    const [instruction,     setInstruction]     = useState<string>("Position your face inside the frame");
    const [isValid,         setIsValid]         = useState<boolean>(false);
    const [isCapturing,     setIsCapturing]     = useState<boolean>(false);
    const [captureProgress, setCaptureProgress] = useState<number>(0);

    // Debug overlay — enabled via ?debug=1 in URL
    const debugMode = new URLSearchParams(window.location.search).get("debug") === "1";
    const [debugInfo, setDebugInfo] = useState({ normSize: 0, faceW: 0, faceH: 0, w: 0, h: 0 });

    // Camera flip state
    const [cameras,        setCameras]        = useState<MediaDeviceInfo[]>([]);
    const [activeCamIdx,   setActiveCamIdx]   = useState<number>(0);

    // Upload state
    const [uploadPreview,  setUploadPreview]  = useState<string | null>(null);
    const [uploadFile,     setUploadFile]     = useState<File | null>(null);
    const [uploadError,    setUploadError]    = useState<string>("");
    const [isConverting,   setIsConverting]   = useState<boolean>(false);

    // MediaPipe refs
    const faceMeshRef    = useRef<any>(null);
    const rafRef         = useRef<number>(0);       // replaces MediaPipe Camera utility
    const capturedFrames = useRef<string[]>([]);
    const warmupCount    = useRef<number>(0);
    const TOTAL_FRAMES   = 10;
    const WARMUP_FRAMES  = 3;

    // ── Helpers ───────────────────────────────────────────────────────────────

    const uploadEndpoint  = apiEndpoint.replace(/\/api\/pd\/measure.*$/, "/api/pd/measure");
    const convertEndpoint = apiEndpoint.replace(/\/api\/pd\/measure.*$/, "/api/convert/heif");

    // ── Camera enumerate ──────────────────────────────────────────────────────

    const enumerateCameras = useCallback(async () => {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            let videoCams = devices.filter(d => d.kind === "videoinput");

            // In a cross-origin iframe, browsers return blank labels until camera
            // permission has been granted. If labels are empty, request a brief
            // getUserMedia stream to unlock them, then re-enumerate.
            if (videoCams.length > 0 && videoCams.every(d => d.label === "")) {
                let tempStream: MediaStream | null = null;
                try {
                    tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
                    const devices2 = await navigator.mediaDevices.enumerateDevices();
                    videoCams = devices2.filter(d => d.kind === "videoinput");
                } finally {
                    tempStream?.getTracks().forEach(t => t.stop());
                }
            }

            // Only show Flip when at least one camera is a back/environment camera.
            // Desktop webcams (even on touch-screen laptops) never expose a back camera,
            // so this correctly hides Flip on all desktops while showing it on phones/tablets.
            const hasBackCamera = videoCams.some(d =>
                /back|rear|environment/i.test(d.label)
            );
            setCameras(hasBackCamera ? videoCams : []);
        } catch {
            // silently ignore
        }
    }, []);

    // ── Camera start / stop ───────────────────────────────────────────────────

    const stopCamera = useCallback(() => {
        cancelAnimationFrame(rafRef.current);
        streamRef.current?.getTracks().forEach(t => t.stop());
        streamRef.current = null;
    }, []);

    const startCamera = useCallback(async (camIndex = 0, camList = cameras) => {
        if (!videoRef.current) return;
        stopCamera();
        try {
            const target = camList[camIndex];
            const constraints: MediaStreamConstraints = {
                video: target?.deviceId
                    ? { deviceId: { exact: target.deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
                    : { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "user" },
            };
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            streamRef.current = stream;
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
        } catch {
            setErrorMsg("Camera access denied. Please allow camera permissions and try again.");
            setStep("error");
        }
    }, [cameras, stopCamera]);

    // ── MediaPipe face-mesh ───────────────────────────────────────────────────

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

            // ── 1. SIZE CHECK ──────────────────────────────────────────────────
            const lms  = res.multiFaceLandmarks[0];
            const xs   = lms.map((l: any) => l.x);
            const ys   = lms.map((l: any) => l.y);
            const faceW   = Math.max(...xs) - Math.min(...xs);
            const faceH   = Math.max(...ys) - Math.min(...ys);
            const actualW = videoRef.current?.videoWidth  ?? 1280;
            const actualH = videoRef.current?.videoHeight ?? 720;

            // faceW/faceH are normalised 0-1 fractions of frame width/height.
            // On portrait mobile the frame is narrower than tall, so faceW alone
            // is much larger at the same physical distance than on landscape desktop.
            // Fix: use the face diagonal as a fraction of the frame diagonal —
            // this is orientation-independent (same value portrait or landscape).
            const frameDiag = Math.sqrt(actualW * actualW + actualH * actualH);
            const faceDiag  = Math.sqrt(
                (faceW * actualW) * (faceW * actualW) +
                (faceH * actualH) * (faceH * actualH)
            );
            const normSize = faceDiag / frameDiag;

            (window as any).__pdFaceW    = faceW;
            (window as any).__pdNormSize = normSize;
            (window as any).__pdActualW  = actualW;

            if (debugMode) setDebugInfo({ normSize, faceW, faceH, w: actualW, h: actualH });

            // Calibrated thresholds for normSize (diagonal ratio):
            // arm's length ≈ 0.19–0.28 on both desktop and portrait mobile
            const minFace = 0.13;
            const maxFace = 0.30;

            if (normSize < minFace) {
                setIsValid(false); setInstruction(`Move closer — about an arm's length away`); return;
            }
            if (normSize > maxFace) {
                setIsValid(false); setInstruction(`Too close — move back a little`); return;
            }

            // ── 2. CENTRE-IN-BOX CHECK ─────────────────────────────────────────
            const rawFaceCx = (Math.min(...xs) + Math.max(...xs)) / 2;
            const rawFaceCy = (Math.min(...ys) + Math.max(...ys)) / 2;
            const faceCx = 1 - rawFaceCx; // mirror for CSS scaleX(-1)
            const faceCy = rawFaceCy;

            const tolerance = 0.04;
            const BOX_X_MIN = 0.36 + tolerance;
            const BOX_X_MAX = 0.64 - tolerance;
            const BOX_Y_MIN = 0.135 + tolerance;
            const BOX_Y_MAX = 0.755 - tolerance;

            if (faceCx < BOX_X_MIN) { setIsValid(false); setInstruction("Move your face to the right"); return; }
            if (faceCx > BOX_X_MAX) { setIsValid(false); setInstruction("Move your face to the left");  return; }
            if (faceCy < BOX_Y_MIN) { setIsValid(false); setInstruction("Move your face down");          return; }
            if (faceCy > BOX_Y_MAX) { setIsValid(false); setInstruction("Move your face up");            return; }

            setIsValid(true);
            if (!isCapturing) setInstruction("Hold still — looking good!");
        });

        faceMeshRef.current = fm;

        // ── RAF loop: feed frames to FaceMesh ourselves ───────────────────────
        // We own the stream via startCamera() so we must NOT use win.Camera —
        // that utility calls getUserMedia() internally and always re-opens the
        // front camera, overwriting whatever camera we selected.
        cancelAnimationFrame(rafRef.current);
        const tick = async () => {
            const video = videoRef.current;
            if (video && !video.paused && !video.ended && faceMeshRef.current) {
                await faceMeshRef.current.send({ image: video });
            }
            rafRef.current = requestAnimationFrame(tick);
        };
        rafRef.current = requestAnimationFrame(tick);
    }, [mediapipeBasePath, isCapturing]);

    useEffect(() => {
        enumerateCameras().then(async () => {
            await startCamera(0);
            const interval = setInterval(() => {
                if ((window as any).FaceMesh && (window as any).Camera) {
                    clearInterval(interval);
                    initFaceMesh();
                }
            }, 200);
        });
        return () => stopCamera();
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // ── Camera flip ───────────────────────────────────────────────────────────

    const handleFlipCamera = useCallback(async () => {
        const nextIdx = (activeCamIdx + 1) % cameras.length;
        setActiveCamIdx(nextIdx);
        warmupCount.current = 0;
        setIsValid(false);
        setInstruction("Position your face inside the frame");
        // Stop the RAF loop and current stream before switching
        cancelAnimationFrame(rafRef.current);
        await startCamera(nextIdx, cameras);
        // Restart the RAF feed loop on the new stream
        initFaceMesh();
    }, [activeCamIdx, cameras, startCamera, initFaceMesh]);

    // ── Capture frames ────────────────────────────────────────────────────────

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

    // ── Submit batch frames ───────────────────────────────────────────────────

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

            // apiEndpoint is already /api/pd/measure-batch (the default)
            const resp = await fetch(apiEndpoint, { method: "POST", body: form });
            if (!resp.ok) throw new Error(`Server error ${resp.status}`);

            const data: PDMeasurementResult = await resp.json();
            setResult(data);
            setStep("results");
            // Pass middle frame as best frame to parent
            const bestFrame = capturedFrames.current[Math.floor(capturedFrames.current.length / 2)] ?? null;
            onMeasurement?.({ ...data, best_frame_dataurl: bestFrame });
        } catch (err: any) {
            const msg = err?.message ?? "Measurement failed. Please try again.";
            setErrorMsg(msg);
            setStep("error");
            onError?.(msg);
        }
    }, [apiEndpoint, onMeasurement, onError, stopCamera]);

    // ── Upload: file select ───────────────────────────────────────────────────

    const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setUploadError("");
        setUploadPreview(null);
        setUploadFile(null);

        const isHeic =
            file.type === "image/heic" ||
            file.type === "image/heif" ||
            /\.(heic|heif)$/i.test(file.name);

        if (isHeic) {
            // Send to backend HEIC→JPEG converter
            setIsConverting(true);
            try {
                const form = new FormData();
                form.append("image", file, file.name);
                const res = await fetch(convertEndpoint, { method: "POST", body: form });
                if (!res.ok) throw new Error("HEIC conversion failed");
                const blob = await res.blob();
                const converted = new File(
                    [blob],
                    file.name.replace(/\.(heic|heif)$/i, ".jpg"),
                    { type: "image/jpeg" }
                );
                setUploadFile(converted);
                setUploadPreview(URL.createObjectURL(blob));
            } catch {
                setUploadError("Could not convert HEIC file. Please try a JPEG or PNG.");
            } finally {
                setIsConverting(false);
            }
        } else {
            setUploadFile(file);
            setUploadPreview(URL.createObjectURL(file));
        }
    }, [convertEndpoint]);

    // ── Upload: submit single image ───────────────────────────────────────────

    const submitUpload = useCallback(async () => {
        if (!uploadFile) return;
        setStep("processing");

        try {
            const form = new FormData();
            form.append("image", uploadFile, uploadFile.name);
            form.append("age_group", "auto");

            const resp = await fetch(uploadEndpoint, { method: "POST", body: form });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail ?? `Server error ${resp.status}`);
            }

            const data: PDMeasurementResult = await resp.json();
            setResult(data);
            setStep("results");
            // For upload mode, use the preview image as the best frame
            onMeasurement?.({ ...data, best_frame_dataurl: uploadPreview ?? null });
        } catch (err: any) {
            const msg = err?.message ?? "Measurement failed. Please try again.";
            setErrorMsg(msg);
            setStep("error");
            onError?.(msg);
        }
    }, [uploadFile, uploadEndpoint, onMeasurement, onError, uploadPreview]);

    // ── Reset ──────────────────────────────────────────────────────────────────

    const handleRestart = useCallback(() => {
        setResult(null);
        setErrorMsg("");
        setUploadPreview(null);
        setUploadFile(null);
        setUploadError("");
        setCaptureProgress(0);
        setIsCapturing(false);
        setIsValid(false);
        warmupCount.current    = 0;
        capturedFrames.current = [];
        setInstruction("Position your face inside the frame");
        // Set step FIRST so React remounts the <video> element,
        // then start camera after the DOM has updated.
        setStep("capture");
        setTimeout(async () => {
            await startCamera(activeCamIdx, cameras);
            initFaceMesh();
        }, 50);
    }, [startCamera, initFaceMesh, activeCamIdx, cameras]);

    const switchToUpload = useCallback(() => {
        stopCamera();
        setUploadPreview(null);
        setUploadFile(null);
        setUploadError("");
        setIsValid(false);
        setInstruction("Position your face inside the frame");
        setStep("upload");
    }, [stopCamera]);

    const switchToCamera = useCallback(() => {
        setUploadPreview(null);
        setUploadFile(null);
        setUploadError("");
        warmupCount.current = 0;
        // Set step first so React remounts the <video> element,
        // then start camera after the DOM has updated.
        setStep("capture");
        setTimeout(async () => {
            await startCamera(activeCamIdx, cameras);
            initFaceMesh();
        }, 50);
    }, [startCamera, initFaceMesh, activeCamIdx, cameras]);

    // ── Render ─────────────────────────────────────────────────────────────────

    return (
        <div className="ifu">

            {/* ── Close button ── */}
            <button
                className="ifu__close-btn"
                aria-label="Close"
                onClick={() => {
                    stopCamera();
                    if (window.parent !== window) {
                        window.parent.postMessage({ type: "PD_CLOSE" }, "*");
                    }
                }}
            >✕</button>

            {/* ── CAPTURE STEP ── */}
            {(step === "capture" || (step === "error" && !uploadFile)) && (
                <div className="ifu__camera-wrap">
                    {/* Live video */}
                    <video ref={videoRef} className="ifu__video" playsInline muted autoPlay />

                    {/* Hidden canvas */}
                    <canvas ref={canvasRef} style={{ display: "none" }} />

                    {/* Face guide overlay */}
                    <div className="ifu__overlay">
                        <div className={`ifu__guide-box ${isValid ? "ifu__guide-box--ok" : ""} ${isCapturing ? "ifu__guide-box--capturing" : ""}`} />

                        {/* Debug overlay — visible only when ?debug=1 */}
                        {debugMode && (
                            <div style={{
                                position: "absolute", top: 8, left: 8, zIndex: 99,
                                background: "rgba(0,0,0,0.72)", color: "#0f0", fontFamily: "monospace",
                                fontSize: 13, padding: "6px 10px", borderRadius: 6, lineHeight: 1.7,
                                pointerEvents: "none",
                            }}>
                                <div>normSize: <b>{debugInfo.normSize.toFixed(4)}</b> (min 0.19 / max 0.32)</div>
                                <div>faceW: {debugInfo.faceW.toFixed(4)} · faceH: {debugInfo.faceH.toFixed(4)}</div>
                                <div>frame: {debugInfo.w}×{debugInfo.h}</div>
                            </div>
                        )}

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
                                <span>{(step === "error" ? errorMsg : null) || instruction}</span>
                            )}
                        </div>

                        {isCapturing && <div className="ifu__badge">LIVE SAMPLING</div>}
                    </div>

                    {/* Bottom toolbar — measure + upload + flip */}
                    {!isCapturing && (
                        <div className="ifu__toolbar">
                            {/* Upload button */}
                            <button className="ifu__tool-btn" onClick={switchToUpload} title="Upload a photo instead">
                                <UploadIcon />
                                <span>Upload</span>
                            </button>

                            {/* Measure PD — primary CTA */}
                            <button
                                className={`ifu__btn ifu__btn--capture ${isValid ? "ifu__btn--ready" : "ifu__btn--waiting"}`}
                                onClick={captureFrames}
                                disabled={!isValid}
                            >
                                <span className="ifu__btn-icon"><EyeIcon /></span>
                                {isValid ? "Measure PD" : "Waiting for face…"}
                            </button>

                            {/* Flip camera — only shown if >1 camera available */}
                            {cameras.length > 1 ? (
                                <button className="ifu__tool-btn" onClick={handleFlipCamera} title="Switch camera">
                                    <FlipCameraIcon />
                                    <span>Flip</span>
                                </button>
                            ) : (
                                /* spacer to keep layout balanced */
                                <div className="ifu__tool-btn ifu__tool-btn--spacer" />
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* ── UPLOAD STEP ── */}
            {step === "upload" && (
                <div className="ifu__upload-wrap">
                    {/* Hidden file input */}
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif,image/*"
                        style={{ display: "none" }}
                        onChange={handleFileSelect}
                    />

                    {isConverting ? (
                        <div className="ifu__upload-converting">
                            <div className="ifu__spinner" />
                            <p>Converting HEIC image…</p>
                        </div>
                    ) : uploadPreview ? (
                        /* Preview + confirm */
                        <div className="ifu__upload-preview-wrap">
                            <img src={uploadPreview} className="ifu__upload-preview" alt="Preview" />
                            <div className="ifu__upload-actions">
                                <button className="ifu__btn ifu__btn--save" onClick={submitUpload}>
                                    <span className="ifu__btn-icon"><EyeIcon /></span>
                                    Measure PD
                                </button>
                                <button className="ifu__btn ifu__btn--restart" onClick={() => {
                                    setUploadPreview(null);
                                    setUploadFile(null);
                                    if (fileInputRef.current) fileInputRef.current.value = "";
                                }}>
                                    Choose different photo
                                </button>
                                <button className="ifu__btn ifu__btn--ghost" onClick={switchToCamera}>
                                    Use camera instead
                                </button>
                            </div>
                            {uploadError && <p className="ifu__upload-error">{uploadError}</p>}
                        </div>
                    ) : (
                        /* Drop zone */
                        <div className="ifu__upload-zone" onClick={() => fileInputRef.current?.click()}>
                            <div className="ifu__upload-icon"><UploadIcon /></div>
                            <p className="ifu__upload-title">Upload a photo</p>
                            <p className="ifu__upload-sub">JPEG, PNG, WebP or HEIC (iPhone)<br />Look straight at the camera in the photo</p>
                            <button className="ifu__btn ifu__btn--ready" style={{ marginTop: 16 }}>
                                Choose photo
                            </button>
                            {uploadError && <p className="ifu__upload-error">{uploadError}</p>}
                            <button className="ifu__upload-back" onClick={(e) => { e.stopPropagation(); switchToCamera(); }}>
                                ← Back to camera
                            </button>
                        </div>
                    )}
                </div>
            )}

            {/* ── PROCESSING STEP ── */}
            {step === "processing" && (
                <div className="ifu__processing">
                    <div className="ifu__spinner" />
                    <p className="ifu__processing-label">Measuring pupil distance…</p>
                    <p className="ifu__processing-sub">Analysing your photo</p>
                </div>
            )}

            {/* ── RESULTS STEP ── */}
            {step === "results" && result && (
                <div className="ifu__results">
                    <div className="ifu__results-bg" />
                    <div className="ifu__results-card">
                        <div className="ifu__results-icon"><EyeIcon /></div>
                        <p className="ifu__results-label">Your pupillary distance is:</p>
                        <p className="ifu__results-value">{result.overall_pd_mm} mm</p>
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
                        <p className="ifu__results-confidence">
                            Confidence: {(result.confidence_score * 100).toFixed(0)}% &nbsp;·&nbsp; ±{result.error_margin.value_mm.toFixed(1)} mm
                        </p>
                        {result.asymmetry_warning && (
                            <p className="ifu__results-warning">
                                ⚠️ Larger-than-usual L/R difference. Consider re-measuring.
                            </p>
                        )}
                    </div>
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
