import React, { useState, useRef, useCallback, useEffect } from "react";
// Type-only imports to avoid bundling issues
import type { Results } from "@mediapipe/face_mesh";
import type { PDMeasurementResult, PDMeasurerProps } from "../../types";
import "./styles.css";

// Access MediaPipe via Global Script Tags (Loaded in index.html)
const FaceMesh = (window as any).FaceMesh;
const Camera = (window as any).Camera;


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

export const PDMeasurer: React.FC<PDMeasurerProps> = ({ 
    apiEndpoint = DEFAULT_API_URL, 
    onMeasurement, 
    onError, 
    className = "", 
    primaryColor, 
    mediapipeBasePath = "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619" 
}) => {
    const [step, setStep] = useState<Step>("capture");
    const [captureMode, setCaptureMode] = useState<CaptureMode>("camera");
    const [referenceType, setReferenceType] = useState<string>("none");
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
    const prevPreviewRef = useRef<string | null>(null);
    const captureAbortRef = useRef<AbortController | null>(null);
    const startCameraRef = useRef<() => Promise<void>>();
    const isComponentMounted = useRef(true);
    const initializationLock = useRef(false);
    const stateRef = useRef({ isAutoCapturing, step, captureMode });
    
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
        console.log("Stopping camera resources...");
        
        if (cameraRef.current) {
            try {
                await cameraRef.current.stop();
            } catch (e) {
                console.error("Error stopping camera utility:", e);
            }
            cameraRef.current = null;
        }

        if (faceMeshRef.current) {
            try {
                await faceMeshRef.current.close();
            } catch (e) {
                console.error("Error closing FaceMesh:", e);
            }
            faceMeshRef.current = null;
        }

        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => {
                track.stop();
                console.log(`Track ${track.label} released.`);
            });
            videoRef.current.srcObject = null;
        }
    }, []);

    const handleMultiFrameSubmit = useCallback(async () => {
        setStep("processing");

        try {
            const formData = new FormData();
            
            // 1. If we have a buffer (Camera mode), use it
            if (captureBufferRef.current.length > 0) {
                captureBufferRef.current.forEach((blob, idx) => {
                    formData.append("images", blob, `frame_${idx}.jpg`);
                });
            } 
            // 2. If no buffer but we have an uploaded file (Upload mode)
            else if (imageFile) {
                formData.append("images", imageFile, imageFile.name);
            }
            else {
                throw new Error("No image data to process.");
            }

            formData.append("reference_type", referenceType);

            // Use the batch endpoint for both single and multi-frame consistency
            const batchEndpoint = apiEndpoint.includes("/api/pd/measure") 
                ? apiEndpoint.replace("/api/pd/measure", "/api/pd/measure-batch")
                : `${apiEndpoint.replace(/\/$/, "")}/batch`;

            const response = await fetch(batchEndpoint, { method: "POST", body: formData });
            
            if (response.ok) {
                const finalResult: PDMeasurementResult = await response.json();
                setResult(finalResult);
                if (onMeasurement) onMeasurement(finalResult);
                setStep("results");
            } else {
                const errData = await response.json();
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

        const totalFrames = 10;
        for (let i = 0; i < totalFrames; i++) {
            if (abort.signal.aborted || !isComponentMounted.current) return;

            if (videoRef.current) {
                const canvas = document.createElement("canvas");
                canvas.width = videoRef.current.videoWidth;
                canvas.height = videoRef.current.videoHeight;
                const ctx = canvas.getContext("2d");
                ctx?.drawImage(videoRef.current, 0, 0);
                
                await new Promise<void>((resolve) => {
                    canvas.toBlob((blob) => {
                        if (blob) captureBufferRef.current.push(blob);
                        resolve();
                    }, "image/jpeg", 0.9);
                });
                setCaptureProgress(i + 1);
            }
            // Faster sampling for better "continuous" feel (75ms instead of 100ms)
            await new Promise(r => setTimeout(r, 75));
        }

        if (!abort.signal.aborted) {
            stopCamera();
            handleMultiFrameSubmit();
        }
    }, [handleMultiFrameSubmit, stopCamera]);

    const onResults = useCallback((results: Results) => {
        const { isAutoCapturing: currentAutoCapture, step: currentStep, captureMode: currentMode } = stateRef.current;

        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            setValidation({ isValid: false, message: "No face detected" });
            return;
        }

        const landmarks = results.multiFaceLandmarks[0];
        
        // 1. Distance Check (Strict Gating)
        const xCoords = landmarks.map(l => l.x);
        const faceWidth = Math.max(...xCoords) - Math.min(...xCoords);
        if (faceWidth < 0.42) { setValidation({ isValid: false, message: "Move closer (Arm's length)" }); return; }
        if (faceWidth > 0.58) { setValidation({ isValid: false, message: "Too close! Move back slightly" }); return; }

        // 2. Horizontal Pose (Yaw) - Tightened to 0.45-0.55 range
        const leftEyeX = landmarks[33].x;
        const rightEyeX = landmarks[263].x;
        const noseX = landmarks[1].x;
        const horizontalSymmetry = (noseX - leftEyeX) / (rightEyeX - leftEyeX);
        if (horizontalSymmetry < 0.45 || horizontalSymmetry > 0.55) {
            setValidation({ isValid: false, message: "Center your face (Yaw)" });
            return;
        }

        // 3. Vertical Pose (Pitch) - Tightened to 0.40-0.60
        const noseY = landmarks[1].y;
        const eyeAvgY = (landmarks[33].y + landmarks[263].y) / 2;
        const mouthAvgY = (landmarks[61].y + landmarks[291].y) / 2;
        const verticalSymmetry = (noseY - eyeAvgY) / (mouthAvgY - eyeAvgY);
        if (verticalSymmetry < 0.40 || verticalSymmetry > 0.60) {
            setValidation({ isValid: false, message: "Look straight (Pitch)" });
            return;
        }

        // 4. Head Rotation (Roll) - Check if eyes are on the same level
        const eyeLevelDiff = Math.abs(landmarks[33].y - landmarks[263].y);
        if (eyeLevelDiff > 0.02) {
            setValidation({ isValid: false, message: "Level your head (Roll)" });
            return;
        }

        // 5. Eyes Open Check (Strict Gating)
        const leftEyeOpen = Math.abs(landmarks[159].y - landmarks[145].y);
        const rightEyeOpen = Math.abs(landmarks[386].y - landmarks[374].y);
        if (leftEyeOpen < 0.008 || rightEyeOpen < 0.008) {
            setValidation({ isValid: false, message: "Keep eyes wide open" });
            return;
        }

        // --- SUCCESS: ALL CHECKS PASS ---
        setValidation({ isValid: true, message: "Perfect! Hold still..." });
        
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
                
                // Set options and results handler
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

                // Test if the library can actually load its assets
                // We don't want to wait for the first frame to discover the CDN is down
                if (currentBasePath.includes("jsdelivr.net")) {
                    console.log("Checking CDN availability...");
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 3000); // 3s timeout
                    
                    try {
                        await fetch(`${currentBasePath}/face_mesh_solution_simd_wasm_bin.wasm`, { 
                            method: 'HEAD', 
                            signal: controller.signal,
                            mode: 'no-cors' 
                        });
                        clearTimeout(timeoutId);
                        // in no-cors mode, we can't check .ok, but if fetch didn't throw, we assume it's reachable
                    } catch (e) {
                        console.warn("CDN unreachable, falling back to local assets.");
                        currentBasePath = "/mediapipe/face_mesh";
                        faceMesh = initFaceMesh(currentBasePath);
                        // Re-apply options for the new instance
                        faceMesh.setOptions({
                            maxNumFaces: 1,
                            refineLandmarks: true,
                            minDetectionConfidence: 0.5,
                            minTrackingConfidence: 0.5,
                        });
                        faceMesh.onResults((results: Results) => {
                            if (!isComponentMounted.current) return;
                            onResults(results);
                        });
                    }
                }
            } catch (e) {
                console.error("Critical failure during FaceMesh initialization:", e);
                // Last ditch effort: Try local if not already tried
                if (currentBasePath !== "/mediapipe/face_mesh") {
                    currentBasePath = "/mediapipe/face_mesh";
                    faceMesh = initFaceMesh(currentBasePath);
                } else {
                    throw e;
                }
            }

            faceMeshRef.current = faceMesh;

            // 4. Initialize and Start Camera
            const camera = new Camera(videoRef.current, {
                onFrame: async () => {
                    const video = videoRef.current;
                    const faceMesh = faceMeshRef.current;
                    
                    if (video && faceMesh && document.visibilityState === "visible" && video.videoWidth > 0) {
                        try {
                            await faceMesh.send({ image: video });
                        } catch (e) {
                            // Suppress errors during transitions
                        }
                    }
                },
                width: 720,
                height: 960,
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
            if (prevPreviewRef.current) {
                URL.revokeObjectURL(prevPreviewRef.current);
            }
        };
    }, [captureMode, step, startCamera, stopCamera]);

    const switchToUpload = () => {
        stopCamera();
        setCaptureMode("upload");
        setStep("capture");
        if (prevPreviewRef.current) {
            URL.revokeObjectURL(prevPreviewRef.current);
            prevPreviewRef.current = null;
        }
        setImageFile(null);
        setImagePreview(null);
        setResult(null);
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            if (prevPreviewRef.current) {
                URL.revokeObjectURL(prevPreviewRef.current);
            }
            const newUrl = URL.createObjectURL(file);
            prevPreviewRef.current = newUrl;
            setImagePreview(newUrl);
            setImageFile(file);
        }
    };

    const handleReset = () => {
        stopCamera();
        setStep("capture");
        setResult(null);
        if (prevPreviewRef.current) {
            URL.revokeObjectURL(prevPreviewRef.current);
            prevPreviewRef.current = null;
        }
        setImageFile(null);
        setImagePreview(null);
        setError(null);
        setIsAutoCapturing(false);
        setCaptureProgress(0);
        setCaptureMode("camera");
        startCamera();
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
                                    <input id="file-input" type="file" accept="image/*" onChange={handleFileSelect} style={{ display: "none" }} />
                                    
                                    <div className="pd-measurer__ref-selector-wrap" onClick={(e) => e.stopPropagation()}>
                                        <label htmlFor="ref-select">Reference Object:</label>
                                        <select 
                                            id="ref-select"
                                            value={referenceType} 
                                            onChange={(e) => setReferenceType(e.target.value)}
                                            className="pd-measurer__ref-select"
                                        >
                                            <option value="none">No reference (Iris Estimation)</option>
                                            <option value="credit_card">Credit Card (Standard)</option>
                                            <option value="coin_gbp_1p">British 1p Coin</option>
                                            <option value="coin_gbp_2p">British 2p Coin</option>
                                            <option value="coin_gbp_5p">British 5p Coin</option>
                                            <option value="coin_gbp_10p">British 10p Coin</option>
                                            <option value="coin_gbp_20p">British 20p Coin</option>
                                            <option value="coin_gbp_50p">British 50p Coin</option>
                                            <option value="coin_gbp_1">British £1 Coin</option>
                                            <option value="coin_gbp_2">British £2 Coin</option>
                                            <option value="ruler">Standard Ruler</option>
                                        </select>
                                    </div>
                                </div>
                            )}

                            {captureMode === "camera" && (
                                <>
                                    <video ref={videoRef} className="pd-measurer__camera-video" playsInline muted />
                                    <div className="pd-measurer__overlay">
                                        <div className={`pd-measurer__guide-box ${validation.isValid ? "pd-measurer__guide-box--active" : ""}`} />
                                        <div className="pd-measurer__instruction-box">
                                            <p className={`pd-measurer__instruction-text ${validation.isValid ? "pd-measurer__instruction-text--success" : ""}`}>
                                                {isAutoCapturing ? `Sampling Data: ${captureProgress}/10` : validation.message}
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
                                        
                                        {!isAutoCapturing && (
                                            <div className="pd-measurer__ref-floating-selector">
                                                <select 
                                                    value={referenceType} 
                                                    onChange={(e) => setReferenceType(e.target.value)}
                                                    className="pd-measurer__ref-select-mini"
                                                >
                                                    <option value="none">No ref (Iris)</option>
                                                    <option value="credit_card">Credit Card</option>
                                                    <option value="ruler">Ruler</option>
                                                </select>
                                            </div>
                                        )}
                                    </div>
                                </>
                            )}

                            {imagePreview && (
                                <div className="pd-measurer__preview">
                                    <img src={imagePreview} alt="Preview" className="pd-measurer__preview-image" />
                                    <button className="pd-measurer__preview-remove" onClick={handleReset}><CloseIcon /></button>
                                </div>
                            )}
                        </div>

                        {captureMode === "camera" && (
                            <button className="pd-measurer__btn pd-measurer__btn--secondary" onClick={switchToUpload}>Switch to Upload Photo</button>
                        )}

                        {captureMode === "upload" && !imageFile && (
                            <button className="pd-measurer__btn pd-measurer__btn--primary" onClick={startCamera}>Switch to Live Camera</button>
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
                        </div>

                        <p className="pd-measurer__disclaimer-text" style={{ fontSize: '0.8rem', opacity: 0.7 }}>{result.disclaimer}</p>
                        <button className="pd-measurer__btn pd-measurer__btn--secondary" onClick={handleReset} style={{ zIndex: 100, position: 'relative' }}>Measure Again</button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default PDMeasurer;
