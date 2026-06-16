import React, { useState, useRef, useCallback, useEffect } from "react";
// Type-only imports to avoid bundling issues
import type { Results } from "@mediapipe/face_mesh";
import type { PDMeasurementResult, PDMeasurerProps } from "../../types";
import "./styles.css";

// Access MediaPipe via Global Script Tags (Loaded in index.html)
const FaceMesh = (window as any).FaceMesh;
const Camera = (window as any).Camera;

// Guard: surface a clear error if MediaPipe scripts haven't loaded yet.
// This catches misconfigured CDN URLs or slow networks before the first user interaction.
if (typeof FaceMesh === "undefined" || typeof Camera === "undefined") {
    console.error(
        "[PDMeasurer] MediaPipe FaceMesh and/or Camera utility not found on window. " +
        "Ensure the MediaPipe CDN scripts are loaded in the HTML <head> before this bundle."
    );
}


// --- SVG Icons ---
const UploadIcon = () => (
    <svg className="pd-measurer__capture-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
);

const CloseIcon = () => (
    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
);

// --- Types ---
type Step = "capture" | "processing" | "results";
type CaptureMode = "upload" | "camera";

interface ValidationState {
    isValid: boolean;
    message: string;
}

const DEFAULT_API_URL = `${import.meta.env.VITE_API_BASE_URL || ""}/api/pd/measure`.replace(/^\/\//, "/");

// ---------------------------------------------------------------------------
// Convert a Blob/File to a base64 data URL using FileReader.
// Data URLs are self-contained strings — unlike blob: URLs they cannot be
// revoked, never go stale, and work correctly inside Shadow DOM or when the
// page is embedded across different origins/ports.
// ---------------------------------------------------------------------------
function blobToDataUrl(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload  = () => resolve(reader.result as string);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
    });
}

export const PDMeasurer: React.FC<PDMeasurerProps> = ({
    apiEndpoint = DEFAULT_API_URL,
    onMeasurement,
    onError,
    className = "",
    primaryColor,
    mediapipeBasePath = "/mediapipe/face_mesh"
}) => {
    const [step, setStep] = useState<Step>("capture");
    const [captureMode, setCaptureMode] = useState<CaptureMode>("camera");
    const referenceType = "none";
    const ageGroup = "auto";
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [result, setResult] = useState<PDMeasurementResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [validation, setValidation] = useState<ValidationState>({ isValid: false, message: "Initializing camera..." });
    const [isAutoCapturing, setIsAutoCapturing] = useState(false);
    const [captureProgress, setCaptureProgress] = useState(0); // 0 to 10

    const containerRef = useRef<HTMLDivElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const faceMeshRef = useRef<any>(null);
    const cameraRef = useRef<any>(null);
    const captureBufferRef = useRef<Blob[]>([]);
    const captureAbortRef = useRef<AbortController | null>(null);
    const startCameraRef = useRef<() => Promise<void>>();
    const isComponentMounted = useRef(true);
    const initializationLock = useRef(false);
    // Guards double-close: once stopCamera() begins tearing down FaceMesh we
    // set this to true so any concurrent onFrame / second teardown call skips it.
    const faceMeshClosingRef = useRef(false);
    const stateRef = useRef({ isAutoCapturing, step, captureMode });
    // Off-screen canvas reused across all frame captures (C2).
    // Created once in startCamera with willReadFrequently hint for fast toBlob.
    const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const captureCtxRef    = useRef<CanvasRenderingContext2D | null>(null);
    // Raw getUserMedia stream — held separately so we can stop its tracks on
    // teardown even if the MediaPipe Camera utility has already released them (C1).
    const rawStreamRef = useRef<MediaStream | null>(null);
    
    // Apply primary color
    useEffect(() => {
        if (primaryColor && containerRef.current) {
            containerRef.current.style.setProperty("--pd-primary", primaryColor);
        }
    }, [primaryColor]);
    
    useEffect(() => {
        stateRef.current = { isAutoCapturing, step, captureMode };
    }, [isAutoCapturing, step, captureMode]);

    // --- Core Logic Functions (Ordered for Dependency Management) ---

    const stopCamera = useCallback(async () => {
        // Guard: if already in the middle of teardown, skip to avoid double-close
        if (faceMeshClosingRef.current) return;
        faceMeshClosingRef.current = true;

        console.log("Stopping camera resources...");

        if (cameraRef.current) {
            try {
                await cameraRef.current.stop();
            } catch (e) {
                console.error("Error stopping camera utility:", e);
            }
            cameraRef.current = null;
        }

        // Null out the ref BEFORE calling close() so any in-flight onFrame
        // that checks faceMeshRef.current will see null and skip send().
        const fm = faceMeshRef.current;
        faceMeshRef.current = null;
        if (fm) {
            try {
                await fm.close();
            } catch (e) {
                // "SolutionWasm instance already deleted" can fire when the WASM
                // runtime tears itself down concurrently — safe to ignore.
                console.warn("FaceMesh close (ignored):", (e as Error).message);
            }
        }

        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => {
                track.stop();
                console.log(`Track ${track.label} released.`);
            });
            videoRef.current.srcObject = null;
        }

        // C1: also stop the raw getUserMedia stream we hold separately.
        // MediaPipe Camera.stop() may not always release every track on Safari.
        if (rawStreamRef.current) {
            rawStreamRef.current.getTracks().forEach(t => t.stop());
            rawStreamRef.current = null;
        }

        // C2: discard the off-screen canvas so it's re-created fresh next time.
        captureCanvasRef.current = null;
        captureCtxRef.current    = null;

        faceMeshClosingRef.current = false;
    }, []);

    const handleMultiFrameSubmit = useCallback(async () => {
        setStep("processing");

        try {
            const formData = new FormData();

            // Upload mode — single image, use /api/pd/measure directly
            if (captureMode === "upload") {
                if (!imageFile) throw new Error("No image selected.");
                formData.append("image", imageFile, imageFile.name);
                formData.append("reference_type", referenceType);
                formData.append("age_group", ageGroup);

                const res = await fetch(apiEndpoint, {
                    method: "POST",
                    body: formData,
                });

                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail ?? `Server error ${res.status}`);
                }

                const data: PDMeasurementResult = await res.json();
                setResult(data);
                setStep("results");
                onMeasurement?.(data);
                return;
            }

            // Camera mode — use captured frame buffer with /api/pd/measure-batch
            if (captureBufferRef.current.length === 0) {
                throw new Error("No frames captured.");
            }

            captureBufferRef.current.forEach((blob, idx) => {
                formData.append("images", blob, `frame_${idx}.png`);
            });

            formData.append("reference_type", referenceType);
            formData.append("age_group", ageGroup);

            // Use the batch endpoint for both single and multi-frame consistency
            const batchEndpoint = apiEndpoint.includes("/api/pd/measure") 
                ? apiEndpoint.replace("/api/pd/measure", "/api/pd/measure-batch")
                : `${apiEndpoint.replace(/\/$/, "")}/batch`;

            // Retry once on transient network failures before surfacing the error
            let response: Response | null = null;
            let lastError: Error | null = null;
            for (let attempt = 0; attempt < 2; attempt++) {
                try {
                    response = await fetch(batchEndpoint, { method: "POST", body: formData });
                    break; // success — exit retry loop
                } catch (networkErr: any) {
                    lastError = networkErr;
                    if (attempt === 0) {
                        // Brief pause before retry
                        await new Promise(r => setTimeout(r, 800));
                    }
                }
            }

            if (!response) {
                throw lastError ?? new Error("Network error. Please check your connection.");
            }

            if (response.ok) {
                const finalResult: PDMeasurementResult = await response.json();
                setResult(finalResult);
                if (onMeasurement) onMeasurement(finalResult);
                setStep("results");
            } else {
                const errData = await response.json().catch(() => ({ detail: null }));
                throw new Error(errData.detail || "Processing failed. Please stay still.");
            }
        } catch (err: any) {
            const errorMessage = err.message || "Quality check failed. Please look straight and try again.";
            setError(errorMessage);
            if (onError) onError(errorMessage);
            setStep("capture");
            setIsAutoCapturing(false);
            setCaptureProgress(0);
            if (captureMode === "camera") startCameraRef.current?.();
        }
    }, [apiEndpoint, captureMode, imageFile, onMeasurement, onError, referenceType]);

    const triggerAutoCapture = useCallback(async () => {
        const abort = new AbortController();
        captureAbortRef.current = abort;

        setIsAutoCapturing(true);
        setCaptureProgress(0);
        captureBufferRef.current = [];

        // Warm-up: skip the first 3 frames after capture starts.
        // The camera's auto-exposure and auto-white-balance algorithms need
        // a few frames to settle after the stream begins — these first frames
        // are consistently darker/flicker-y and produce low blur scores.
        const WARMUP_FRAMES = 3;
        for (let w = 0; w < WARMUP_FRAMES; w++) {
            await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Smart capture: attempt up to MAX_ATTEMPTS frames to collect TARGET_GOOD good ones.
        // A "good" frame is one where the face passes all frontend checks (is_valid).
        // This mirrors the backend's blink/blur/yaw gating — frames that fail here
        // will almost certainly be rejected on the backend too, so skipping them early
        // reduces wasted network bandwidth and improves the final measurement.
        const TARGET_GOOD  = 10;
        const MAX_ATTEMPTS = 20;  // allow extra attempts to compensate for blinks / RAF misses
        let goodFrames = 0;

        for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            if (abort.signal.aborted || !isComponentMounted.current) return;

            // Only capture when the face is in a valid position (live check via stateRef)
            const isValid = stateRef.current.isAutoCapturing;
            if (videoRef.current && isValid) {
                const video = videoRef.current;
                const vw = video.videoWidth;
                const vh = video.videoHeight;

                // C2: reuse the off-screen canvas created in startCamera.
                // If for any reason it was cleared (e.g. stopCamera race), re-create it here.
                if (!captureCanvasRef.current || !captureCtxRef.current) {
                    const cv = document.createElement("canvas");
                    cv.width  = vw;
                    cv.height = vh;
                    // C3: willReadFrequently — browser can optimise the backing store for
                    // frequent readback (used internally by toBlob).
                    const cx = cv.getContext("2d", { willReadFrequently: true });
                    captureCanvasRef.current = cv;
                    captureCtxRef.current    = cx;
                } else {
                    // Update dimensions if resolution changed (e.g. camera re-negotiated)
                    if (captureCanvasRef.current.width !== vw) captureCanvasRef.current.width  = vw;
                    if (captureCanvasRef.current.height !== vh) captureCanvasRef.current.height = vh;
                }

                const ctx = captureCtxRef.current;
                if (ctx) {
                    // C4: sync capture to the next actual decoded video frame boundary.
                    // Without this, drawImage can grab a partially-decoded or repeated frame.
                    await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));

                    // C3: disable interpolation — canvas size matches video exactly (1280×960),
                    // so no resampling is needed.  Bilinear smoothing blurs iris edges.
                    ctx.imageSmoothingEnabled = false;
                    ctx.drawImage(video, 0, 0);
                }

                // C5: PNG is lossless — no DCT block artifacts on iris edges.
                // JPEG at 0.9 reduced Laplacian variance 30–50%, killing blur gate scores.
                await new Promise<void>((resolve) => {
                    captureCanvasRef.current!.toBlob((blob) => {
                        if (blob) {
                            captureBufferRef.current.push(blob);
                            goodFrames++;
                        }
                        resolve();
                    }, "image/png");
                });

                setCaptureProgress(goodFrames);

                // Stop early once we have enough good frames
                if (goodFrames >= TARGET_GOOD) break;
            }

            // 400ms between frames — allows natural micro-movement between captures,
            // improving median robustness. Total: ~4-8s for 10-20 frames.
            await new Promise(r => setTimeout(r, 400));
        }

        if (!abort.signal.aborted) {
            stopCamera();
            handleMultiFrameSubmit();
        }
    }, [handleMultiFrameSubmit, stopCamera]);

    const onResults = useCallback((results: Results) => {
        const { isAutoCapturing: currentAutoCapture, step: currentStep, captureMode: currentMode } = stateRef.current;

        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            setValidation({ isValid: false, message: "No face detected — look at the camera" });
            return;
        }

        const landmarks = results.multiFaceLandmarks[0];

        // 1. Distance Check
        // faceWidth is normalised 0–1 of frame width. At higher resolutions (1920px)
        // the same physical distance produces a smaller normalised value than at 1280px.
        // We scale the thresholds relative to a 1280px baseline so arm's-length works
        // on both laptop (1920px) and mobile (720–1280px).
        const xCoords = landmarks.map(l => l.x);
        const faceWidth = Math.max(...xCoords) - Math.min(...xCoords);

        const actualVideoWidth = videoRef.current?.videoWidth || 1280;
        const resolutionScale  = 1280 / actualVideoWidth;   // 1.0 at 1280, 0.667 at 1920, 1.78 at 720
        const MIN_FACE = 0.28 * resolutionScale;  // arm's length: ~0.28 at 1280, ~0.187 at 1920
        const MAX_FACE = 0.65 * resolutionScale;  // too close:    ~0.65 at 1280, ~0.433 at 1920

        if (faceWidth < MIN_FACE) { setValidation({ isValid: false, message: "📏 Move closer — arm's length away" }); return; }
        if (faceWidth > MAX_FACE) { setValidation({ isValid: false, message: "↔️ Too close! Step back slightly" }); return; }

        // 2. Horizontal Pose (Yaw) — T2-6: relaxed from 0.44–0.56 to 0.40–0.60.
        // The backend gate is 0.35–0.65; keeping frontend slightly tighter
        // (~±10° vs backend ±15°) still guides the user toward centre while
        // accepting marginally off-centre faces that the backend can handle.
        const leftEyeX = landmarks[33].x;
        const rightEyeX = landmarks[263].x;
        const noseX = landmarks[1].x;
        const horizontalSymmetry = (noseX - leftEyeX) / (rightEyeX - leftEyeX);
        if (horizontalSymmetry < 0.40 || horizontalSymmetry > 0.60) {
            const dir = horizontalSymmetry < 0.40 ? "⬅️ Turn face right" : "➡️ Turn face left";
            setValidation({ isValid: false, message: dir });
            return;
        }

        // 3. Vertical Pose (Pitch) — T2-6: relaxed from 0.38–0.62 to 0.35–0.65.
        // Backend accepts 0.30–0.70; frontend sits comfortably inside that.
        const noseY = landmarks[1].y;
        const eyeAvgY = (landmarks[33].y + landmarks[263].y) / 2;
        const mouthAvgY = (landmarks[61].y + landmarks[291].y) / 2;
        const verticalSymmetry = (noseY - eyeAvgY) / (mouthAvgY - eyeAvgY);
        if (verticalSymmetry < 0.35 || verticalSymmetry > 0.65) {
            const dir = verticalSymmetry < 0.35 ? "⬆️ Tilt chin down slightly" : "⬇️ Tilt chin up slightly";
            setValidation({ isValid: false, message: dir });
            return;
        }

        // 4. Head Roll — eyes must be level
        const eyeLevelDiff = Math.abs(landmarks[33].y - landmarks[263].y);
        if (eyeLevelDiff > 0.018) {
            setValidation({ isValid: false, message: "↕️ Level your head — don't tilt sideways" });
            return;
        }

        // 5. Eyes open check
        const leftEyeOpen  = Math.abs(landmarks[159].y - landmarks[145].y);
        const rightEyeOpen = Math.abs(landmarks[386].y - landmarks[374].y);
        if (leftEyeOpen < 0.008 || rightEyeOpen < 0.008) {
            setValidation({ isValid: false, message: "👁️ Open your eyes wide" });
            return;
        }

        // 6. Eye-symmetry lighting heuristic — T2-6: relaxed from 0.6 to 0.50.
        // A 0.6 threshold was rejecting faces with slightly asymmetric eyes
        // (common in ~20% of people) even under good lighting. 0.50 still
        // catches heavily uneven lighting (one eye nearly closed) while
        // improving frame acceptance for normal faces.
        const eyeSymmetry = Math.min(leftEyeOpen, rightEyeOpen) / Math.max(leftEyeOpen, rightEyeOpen);
        if (eyeSymmetry < 0.50) {
            setValidation({ isValid: false, message: "💡 Check lighting — one eye looks darker" });
            return;
        }

        // --- ALL CHECKS PASS ---
        setValidation({ isValid: true, message: "✅ Perfect! Hold still..." });

        if (!currentAutoCapture && currentStep === "capture" && currentMode === "camera") {
            triggerAutoCapture();
        }
    }, [triggerAutoCapture]);

    const startCamera = useCallback(async () => {
        if (initializationLock.current) return;
        initializationLock.current = true;

        console.log("Starting camera sequence...");
        setValidation({ isValid: false, message: "Starting camera..." });

        if (!FaceMesh || !Camera) {
            setError("Face detection libraries failed to load. Check connection.");
            return;
        }

        try {
            // 1. Full Teardown first to be safe
            await stopCamera();
            // Reset the closing guard so the new FaceMesh instance can be torn down later
            faceMeshClosingRef.current = false;

            // 2. Wait for DOM/Hardware to settle
            await new Promise(resolve => setTimeout(resolve, 500));
            if (!isComponentMounted.current || !videoRef.current) return;

            // 3. Initialize FaceMesh
            let currentBasePath = mediapipeBasePath;
            let faceMesh;

            const initFaceMesh = (path: string) => {
                return new (FaceMesh as any)({
                    locateFile: (file: string) => `${path}/${file}`,
                });
            };

            try {
                faceMesh = initFaceMesh(currentBasePath);
                
                faceMesh.setOptions({
                    maxNumFaces: 1,
                    refineLandmarks: true,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                });

                faceMesh.onResults((results: Results) => {
                    if (!isComponentMounted.current) return;
                    setValidation(prev => (prev.message === "Starting camera..." || prev.message === "Detecting face..." ? { isValid: false, message: "No face detected" } : prev));
                    onResults(results);
                });
            } catch (e) {
                console.error("Critical failure during FaceMesh initialization:", e);
                setError("Face detection failed to initialize. Please refresh.");
                return;
            }

            faceMeshRef.current = faceMesh;

            // 4. C1: Acquire the camera stream with explicit resolution constraints BEFORE
            //    handing it to MediaPipe Camera.  The MediaPipe Camera utility's width/height
            //    are hints to its internal resizer, not getUserMedia constraints — on Safari
            //    and mobile the browser may silently grant 640×480 instead of 1280×960.
            //    By calling getUserMedia ourselves we lock the resolution in the hardware layer.
            //    M3 MacBook FaceTime HD supports 1920×1080 — ask for it explicitly.
            let stream: MediaStream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width:      { ideal: 1920, min: 1280 },
                        height:     { ideal: 1080, min: 720  },
                        facingMode: { ideal: "user" },
                        frameRate:  { ideal: 30, min: 15 },
                        // resizeMode:"none" tells the browser not to software-resize
                        // the sensor output — eliminates one softening pass that
                        // reduces Laplacian variance.
                        // @ts-ignore — resizeMode is part of the spec but missing from
                        // older TypeScript lib.dom.d.ts
                        resizeMode: "none",
                    },
                    audio: false,
                });
            } catch (mediaErr) {
                console.warn("getUserMedia 1920×1080 failed, trying 1280×720:", mediaErr);
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            width:      { ideal: 1280, min: 640 },
                            height:     { ideal: 720,  min: 480 },
                            facingMode: { ideal: "user" },
                            frameRate:  { ideal: 30, min: 15 },
                        },
                        audio: false,
                    });
                } catch (mediaErr2) {
                    console.warn("getUserMedia 1280×720 failed, using browser default:", mediaErr2);
                    // Last resort: let the browser choose — still better than crashing
                    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
                }
            }
            rawStreamRef.current = stream;

            // Log the actual resolution granted by the browser/hardware
            const videoTrack = stream.getVideoTracks()[0];
            const settings   = videoTrack?.getSettings();
            console.log(`Camera granted: ${settings?.width}×${settings?.height} @ ${settings?.frameRate}fps (resizeMode: ${(settings as any)?.resizeMode ?? "unknown"})`);

            // Attach the stream to the video element so MediaPipe Camera can read it
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                // Wait for loadeddata (first actual frame decoded) not just loadedmetadata
                // so canvas dimensions are always accurate when we pre-warm it.
                await new Promise<void>((resolve) => {
                    if (!videoRef.current) { resolve(); return; }
                    if (videoRef.current.readyState >= 2) { resolve(); return; }
                    videoRef.current.onloadeddata = () => resolve();
                });
            }

            // C2+C3: Pre-warm the off-screen capture canvas once we know the real resolution.
            // Doing this here (before capture starts) means the first captured frame doesn't
            // pay a canvas-creation cost, and willReadFrequently is set on the context early.
            if (videoRef.current && videoRef.current.videoWidth > 0) {
                const vw = videoRef.current.videoWidth;
                const vh = videoRef.current.videoHeight;
                const cv = document.createElement("canvas");
                cv.width  = vw;
                cv.height = vh;
                const cx = cv.getContext("2d", { willReadFrequently: true });
                if (cx) cx.imageSmoothingEnabled = false;
                captureCanvasRef.current = cv;
                captureCtxRef.current    = cx;
                console.log(`Capture canvas pre-warmed at ${vw}×${vh}`);
            }

            // 5. Initialize and Start Camera
            // Pass the actual stream dimensions to MediaPipe Camera so it doesn't
            // internally downscale the frames we fought to get at full resolution.
            const actualW = videoRef.current?.videoWidth  || (settings?.width  ?? 1280);
            const actualH = videoRef.current?.videoHeight || (settings?.height ?? 720);
            const camera = new Camera(videoRef.current, {
                onFrame: async () => {
                    const video = videoRef.current;
                    const faceMesh = faceMeshRef.current;

                    // Bail immediately if FaceMesh is being torn down or already gone
                    if (!faceMesh || faceMeshClosingRef.current) return;

                    if (video && document.visibilityState === "visible" && video.videoWidth > 0) {
                        try {
                            await faceMesh.send({ image: video });
                        } catch (e) {
                            // Suppress errors during transitions
                        }
                    }
                },
                width:  actualW,
                height: actualH,
            });

            cameraRef.current = camera;
            await camera.start();
            console.log("Camera started successfully.");
            setValidation(prev => (prev.message === "Starting camera..." ? { isValid: false, message: "Detecting face..." } : prev));

        } catch (err) {
            console.error("Failed to start camera:", err);
            setError("Camera failed to initialize. Please refresh.");
        } finally {
            initializationLock.current = false;
        }
    }, [mediapipeBasePath, onResults, stopCamera]);

    // Keep the ref updated for handles that need it
    useEffect(() => {
        startCameraRef.current = startCamera;
    }, [startCamera]);

    useEffect(() => {
        isComponentMounted.current = true;
        
        if (captureMode === "camera" && step === "capture") {
            startCamera();
        }

        const handleVisibility = () => {
            if (document.visibilityState === "hidden") {
                console.log("Visibility: Hidden. Cleaning up...");
                stopCamera();
            } else {
                console.log("Visibility: Visible. Re-starting...");
                const { step: s, captureMode: c } = stateRef.current;
                if (s === "capture" && c === "camera") {
                    startCamera();
                }
            }
        };

        document.addEventListener("visibilitychange", handleVisibility);

        return () => {
            isComponentMounted.current = false;
            captureAbortRef.current?.abort();
            document.removeEventListener("visibilitychange", handleVisibility);
            stopCamera();
        };
    }, [captureMode, step, startCamera, stopCamera]);

    const switchToUpload = async () => {
        await stopCamera();
        captureBufferRef.current = []; // clear stale camera frames
        setCaptureMode("upload");
        setStep("capture");
        setImageFile(null);
        setImagePreview(null);
        setResult(null);
    };

    const switchToCamera = async () => {
        setImageFile(null);
        setImagePreview(null);
        setResult(null);
        captureBufferRef.current = [];
        setCaptureMode("camera");
        setStep("capture");
        // startCamera() is triggered by the useEffect that watches captureMode + step
    };

    const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setError(null);

        const isHeif =
            file.type === "image/heic" ||
            file.type === "image/heif" ||
            /\.(heic|heif)$/i.test(file.name);

        try {
            if (!isHeif) {
                // JPEG / PNG / WebP — convert directly to data URL in the browser
                const dataUrl = await blobToDataUrl(file);
                setImagePreview(dataUrl);
                setImageFile(file);
                return;
            }

            // HEIC / HEIF — send raw file to backend for conversion, use
            // the returned JPEG for both preview and upload.
            setImagePreview("converting"); // shows spinner

            const convertUrl = `${apiEndpoint
                .replace(/\/api\/pd\/measure.*$/, "")}/api/convert/heif`;

            const formData = new FormData();
            formData.append("image", file, file.name);

            const res = await fetch(convertUrl, { method: "POST", body: formData });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail ?? `Conversion failed (HTTP ${res.status})`);
            }

            const jpegBlob = await res.blob();
            const jpegFile = new File(
                [jpegBlob],
                file.name.replace(/\.(heic|heif)$/i, ".jpg"),
                { type: "image/jpeg" },
            );

            const dataUrl = await blobToDataUrl(jpegBlob);
            setImagePreview(dataUrl);
            setImageFile(jpegFile);

        } catch (err) {
            console.error("Failed to process image:", err);
            setError(
                err instanceof Error
                    ? err.message
                    : "Could not read this image. Please try a JPEG, PNG, or HEIF photo."
            );
            setImagePreview(null);
        }
    };

    const handleReset = () => {
        // Stay in whichever mode the user was in (camera or upload)
        if (captureMode === "camera") {
            stopCamera();
        }
        captureBufferRef.current = []; // clear stale frames
        setStep("capture");
        setResult(null);
        setImageFile(null);
        setImagePreview(null);
        setError(null);
        setIsAutoCapturing(false);
        setCaptureProgress(0);
        // Camera mode: useEffect re-runs on step→"capture" and calls startCamera()
        // Upload mode: nothing to restart, user picks a new file
    };

    return (
        <div ref={containerRef} className={`pd-measurer ${className}`}>
            <div className="pd-measurer__header">
                <h2 className="pd-measurer__title">Advanced PD Measurement</h2>
            </div>

            <div className="pd-measurer__body">
                <div className="pd-measurer__steps">
                    <div className={`pd-measurer__step ${step === "capture" ? "pd-measurer__step--active" : "pd-measurer__step--completed"}`}>1</div>
                    <div className="pd-measurer__step-divider" />
                    <div className={`pd-measurer__step ${step === "processing" ? "pd-measurer__step--active" : step === "results" ? "pd-measurer__step--completed" : "pd-measurer__step--pending"}`}>2</div>
                    <div className="pd-measurer__step-divider" />
                    <div className={`pd-measurer__step ${step === "results" ? "pd-measurer__step--active" : "pd-measurer__step--pending"}`}>3</div>
                </div>

                {error && <div className="pd-measurer__error"><p>{error}</p></div>}

                {step === "capture" && (
                    <>
                        <div className="pd-measurer__guide-container">
                            {captureMode === "upload" && !imagePreview && (
                                <div className="pd-measurer__capture" onClick={() => document.getElementById("file-input")?.click()}>
                                    <UploadIcon />
                                    <p className="pd-measurer__capture-text">Tap to Upload Photo</p>
                                    <p style={{ fontSize: "0.75rem", opacity: 0.6, margin: "4px 0 8px" }}>
                                        JPEG · PNG · WebP · HEIF/HEIC
                                    </p>
                                    <input
                                        id="file-input"
                                        type="file"
                                        accept="image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif,image/*"
                                        onChange={handleFileSelect}
                                        style={{ display: "none" }}
                                    />
                                </div>
                            )}

                            {captureMode === "camera" && (
                                <>
                                    <video ref={videoRef} className="pd-measurer__camera-video" playsInline muted />
                                    <div className="pd-measurer__overlay">
                                        <div className={`pd-measurer__guide-box ${validation.isValid ? "pd-measurer__guide-box--active" : ""}`} />
                                        <div className="pd-measurer__instruction-box">
                                            <p className={`pd-measurer__instruction-text ${validation.isValid ? "pd-measurer__instruction-text--success" : ""}`}>
                                                {isAutoCapturing ? `Good frames: ${captureProgress}/10 — hold still` : validation.message}
                                            </p>
                                            {isAutoCapturing && (
                                                <div className="pd-measurer__progress-container">
                                                    <div className="pd-measurer__progress-bar" style={{ width: `${(captureProgress / 10) * 100}%` }} />
                                                </div>
                                            )}
                                        </div>
                                        {isAutoCapturing && (
                                            <div className="pd-measurer__sampling-badge">
                                                LIVE SAMPLING
                                            </div>
                                        )}
                                        

                                    </div>
                                </>
                            )}

                            {imagePreview === "converting" && (
                                <div className="pd-measurer__preview" style={{ display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 8 }}>
                                    <div style={{ width: 36, height: 36, border: "4px solid #ccc", borderTop: "4px solid #333", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
                                    <p style={{ fontSize: "0.85rem", opacity: 0.7 }}>Converting HEIF image…</p>
                                </div>
                            )}

                            {imagePreview && imagePreview !== "converting" && (
                                <div className="pd-measurer__preview">
                                    <img
                                        src={imagePreview}
                                        alt="Preview"
                                        className="pd-measurer__preview-image"
                                        style={{ display: "block", width: "100%", height: "100%", objectFit: "cover" }}
                                        onError={(e) => {
                                            // Fallback: show a broken-image message instead of a broken icon
                                            (e.target as HTMLImageElement).style.display = "none";
                                            setError("Preview could not be displayed, but the image will still be processed.");
                                        }}
                                    />
                                    <button className="pd-measurer__preview-remove" onClick={handleReset}><CloseIcon /></button>
                                </div>
                            )}
                        </div>

                        {captureMode === "camera" && (
                            <button className="pd-measurer__btn pd-measurer__btn--secondary" onClick={switchToUpload}>Switch to Upload Photo</button>
                        )}

                        {captureMode === "upload" && !imageFile && (
                            <button className="pd-measurer__btn pd-measurer__btn--primary" onClick={switchToCamera}>Switch to Live Camera</button>
                        )}
                        
                        {imageFile && (
                            <button className="pd-measurer__btn pd-measurer__btn--primary" onClick={handleMultiFrameSubmit}>Measure PD</button>
                        )}
                    </>
                )}

                {step === "processing" && (
                    <div className="pd-measurer__loading">
                        <div className="pd-measurer__spinner" />
                        <p>Processing multi-frame data...</p>
                    </div>
                )}

                {step === "results" && result && (
                    <div className="pd-measurer__results">
                        <h3 className="pd-measurer__results-title">Your PD Results</h3>
                        <div className="pd-measurer__measurements">
                            <div className="pd-measurer__measurement">
                                <div className="pd-measurer__measurement-label">Overall</div>
                                <div className="pd-measurer__measurement-value">{result.overall_pd_mm}mm</div>
                            </div>
                            <div className="pd-measurer__measurement">
                                <div className="pd-measurer__measurement-label">Left</div>
                                <div className="pd-measurer__measurement-value">{result.left_pd_mm}mm</div>
                            </div>
                            <div className="pd-measurer__measurement">
                                <div className="pd-measurer__measurement-label">Right</div>
                                <div className="pd-measurer__measurement-value">{result.right_pd_mm}mm</div>
                            </div>
                        </div>

                        <div className="pd-measurer__stats">
                            <div className="pd-measurer__stat">
                                <span className="pd-measurer__stat-label">Confidence:</span>
                                <span className="pd-measurer__stat-value">{(result.confidence_score * 100).toFixed(0)}%</span>
                            </div>
                            <div className="pd-measurer__stat">
                                <span className="pd-measurer__stat-label">Precision:</span>
                                <span className="pd-measurer__stat-value">±{result.error_margin.value_mm.toFixed(1)}mm</span>
                            </div>
                            {result.frames_accepted != null && (
                                <div className="pd-measurer__stat">
                                    <span className="pd-measurer__stat-label">Frames used:</span>
                                    <span className="pd-measurer__stat-value">
                                        {result.frames_accepted}
                                        {result.frames_rejected ? ` (${result.frames_rejected} filtered)` : ""}
                                    </span>
                                </div>
                            )}
                            {result.age_group_used && (
                                <div className="pd-measurer__stat">
                                    <span className="pd-measurer__stat-label">Age group:</span>
                                    <span className="pd-measurer__stat-value" style={{ textTransform: "capitalize" }}>
                                        {result.age_group_used.replace("_", " ")}
                                        {ageGroup === "auto" ? " (auto)" : ""}
                                    </span>
                                </div>
                            )}
                        </div>

                        {result.asymmetry_warning && (
                            <div className="pd-measurer__warning" style={{
                                background: "#fff3cd",
                                border: "1px solid #ffc107",
                                borderRadius: "6px",
                                padding: "8px 12px",
                                marginTop: "8px",
                                fontSize: "0.8rem",
                                color: "#856404",
                            }}>
                                ⚠️ Left/right difference is larger than usual. Consider retaking or
                                verifying with an optician for prescription eyewear.
                            </div>
                        )}

                        <p className="pd-measurer__disclaimer-text" style={{ fontSize: '0.8rem', opacity: 0.7 }}>{result.disclaimer}</p>
                        <button className="pd-measurer__btn pd-measurer__btn--secondary" onClick={handleReset} style={{ zIndex: 100, position: 'relative' }}>Measure Again</button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default PDMeasurer;
