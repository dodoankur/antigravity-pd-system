import React, { useState, useRef, useCallback } from "react";
import type { ReferenceType, PDMeasurementResult, PDMeasurerProps } from "../../types";
import "./styles.css";

// SVG Icons as components
const UploadIcon = () => (
    <svg className="pd-measurer__capture-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
        />
    </svg>
);

const CameraIcon = () => (
    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
        />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
);

const CheckIcon = () => (
    <svg className="pd-measurer__results-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);

const WarningIcon = () => (
    <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20">
        <path
            fillRule="evenodd"
            d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 011 1v3a1 1 0 11-2 0V6a1 1 0 011-1z"
            clipRule="evenodd"
        />
    </svg>
);

const CloseIcon = () => (
    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
);

// Reference type options
const referenceOptions: { value: ReferenceType; label: string; hint?: string }[] = [
    { value: "none", label: "No Reference (Iris Estimation)", hint: "Uses average iris diameter" },
    { value: "credit_card", label: "Credit Card", hint: "Place card flat next to face" },
    { value: "coin_gbp_1", label: "£1 Coin (23.43mm)", hint: "Hold coin at eye level" },
    { value: "coin_gbp_2", label: "£2 Coin (28.4mm)", hint: "Hold coin at eye level" },
    { value: "coin_gbp_50p", label: "50 Pence (27.3mm)", hint: "Hold coin at eye level" },
    { value: "coin_gbp_20p", label: "20 Pence (21.4mm)", hint: "Hold coin at eye level" },
    { value: "coin_gbp_10p", label: "10 Pence (24.5mm)", hint: "Hold coin at eye level" },
    { value: "ruler", label: "Ruler", hint: "Place ruler at eye level" },
];

type Step = "capture" | "processing" | "results";
type CaptureMode = "upload" | "camera";

export const PDMeasurer: React.FC<PDMeasurerProps> = ({ apiEndpoint = "/api/pd/measure", onMeasurement, onError, className = "", primaryColor }) => {
    const [step, setStep] = useState<Step>("capture");
    const [captureMode, setCaptureMode] = useState<CaptureMode>("upload");
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [referenceType, setReferenceType] = useState<ReferenceType>("none");
    const [result, setResult] = useState<PDMeasurementResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isStreaming, setIsStreaming] = useState(false);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);

    // Apply custom primary color
    const customStyle = primaryColor
        ? ({
              "--pd-primary": primaryColor,
              "--pd-primary-hover": primaryColor,
          } as React.CSSProperties)
        : undefined;

    // Handle file selection
    const handleFileSelect = useCallback((file: File) => {
        if (!file.type.startsWith("image/")) {
            setError("Please select an image file (JPEG, PNG, WebP)");
            return;
        }

        setImageFile(file);
        setError(null);

        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
            setImagePreview(e.target?.result as string);
        };
        reader.readAsDataURL(file);
    }, []);

    // Handle file input change
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            handleFileSelect(file);
        }
    };

    // Handle drag and drop
    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            if (file) {
                handleFileSelect(file);
            }
        },
        [handleFileSelect],
    );

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
    };

    // Start camera
    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
            });
            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
            setIsStreaming(true);
            setError(null);
        } catch (err) {
            setError("Unable to access camera. Please check permissions or use file upload.");
        }
    };

    // Stop camera
    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }
        setIsStreaming(false);
    };

    // Capture from camera
    const captureFromCamera = () => {
        if (!videoRef.current) return;

        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.drawImage(videoRef.current, 0, 0);

        canvas.toBlob(
            (blob) => {
                if (blob) {
                    const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
                    setImageFile(file);
                    setImagePreview(canvas.toDataURL("image/jpeg"));
                    stopCamera();
                }
            },
            "image/jpeg",
            0.9,
        );
    };

    // Remove preview
    const removePreview = () => {
        setImageFile(null);
        setImagePreview(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    // Submit for measurement
    const handleSubmit = async () => {
        if (!imageFile) {
            setError("Please upload or capture an image first");
            return;
        }

        setStep("processing");
        setError(null);

        try {
            const formData = new FormData();
            formData.append("image", imageFile);
            formData.append("reference_type", referenceType);

            const response = await fetch(apiEndpoint, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Measurement failed");
            }

            const data: PDMeasurementResult = await response.json();
            setResult(data);
            setStep("results");
            onMeasurement?.(data);
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : "An error occurred";
            setError(errorMessage);
            onError?.(errorMessage);
            setStep("capture");
        }
    };

    // Reset to start
    const handleReset = () => {
        setStep("capture");
        setImageFile(null);
        setImagePreview(null);
        setResult(null);
        setError(null);
        stopCamera();
    };

    // Get confidence level
    const getConfidenceLevel = (score: number): { label: string; class: string } => {
        if (score >= 0.85) return { label: "High", class: "high" };
        if (score >= 0.7) return { label: "Moderate", class: "moderate" };
        return { label: "Low", class: "low" };
    };

    return (
        <div className={`pd-measurer ${className}`} style={customStyle}>
            <div className="pd-measurer__header">
                <h2 className="pd-measurer__title">Measure Pupil Distance</h2>
            </div>

            <div className="pd-measurer__body">
                {/* Step indicator */}
                <div className="pd-measurer__steps">
                    <div
                        className={`pd-measurer__step ${step === "capture" ? "pd-measurer__step--active" : step === "processing" || step === "results" ? "pd-measurer__step--completed" : "pd-measurer__step--pending"}`}
                    >
                        1
                    </div>
                    <div className="pd-measurer__step-divider" />
                    <div
                        className={`pd-measurer__step ${step === "processing" ? "pd-measurer__step--active" : step === "results" ? "pd-measurer__step--completed" : "pd-measurer__step--pending"}`}
                    >
                        2
                    </div>
                    <div className="pd-measurer__step-divider" />
                    <div className={`pd-measurer__step ${step === "results" ? "pd-measurer__step--active" : "pd-measurer__step--pending"}`}>3</div>
                </div>

                {/* Error display */}
                {error && (
                    <div className="pd-measurer__error">
                        <p className="pd-measurer__error-text">{error}</p>
                    </div>
                )}

                {/* Capture step */}
                {step === "capture" && (
                    <>
                        {/* Capture mode tabs */}
                        <div className="pd-measurer__tabs">
                            <button
                                className={`pd-measurer__tab ${captureMode === "upload" ? "pd-measurer__tab--active" : ""}`}
                                onClick={() => {
                                    setCaptureMode("upload");
                                    stopCamera();
                                }}
                            >
                                Upload Photo
                            </button>
                            <button
                                className={`pd-measurer__tab ${captureMode === "camera" ? "pd-measurer__tab--active" : ""}`}
                                onClick={() => {
                                    setCaptureMode("camera");
                                }}
                            >
                                Use Camera
                            </button>
                        </div>

                        {/* Upload mode */}
                        {captureMode === "upload" && (
                            <>
                                {!imagePreview ? (
                                    <div className="pd-measurer__capture" onClick={() => fileInputRef.current?.click()} onDrop={handleDrop} onDragOver={handleDragOver}>
                                        <UploadIcon />
                                        <p className="pd-measurer__capture-text">Click to upload or drag and drop</p>
                                        <p className="pd-measurer__capture-hint">JPEG, PNG or WebP • Clear, front-facing photo with eyes open</p>
                                        <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} style={{ display: "none" }} />
                                    </div>
                                ) : (
                                    <div className="pd-measurer__preview">
                                        <img src={imagePreview} alt="Preview" className="pd-measurer__preview-image" />
                                        <button className="pd-measurer__preview-remove" onClick={removePreview} title="Remove image">
                                            <CloseIcon />
                                        </button>
                                    </div>
                                )}
                            </>
                        )}

                        {/* Camera mode */}
                        {captureMode === "camera" && (
                            <div className="pd-measurer__camera">
                                {!imagePreview ? (
                                    <>
                                        <video
                                            ref={videoRef}
                                            autoPlay
                                            playsInline
                                            muted
                                            className="pd-measurer__camera-video"
                                            style={{ display: isStreaming ? "block" : "none" }}
                                        />
                                        <div className="pd-measurer__camera-controls">
                                            {!isStreaming ? (
                                                <button className="pd-measurer__btn pd-measurer__btn--primary" onClick={startCamera}>
                                                    <CameraIcon />
                                                    Start Camera
                                                </button>
                                            ) : (
                                                <>
                                                    <button className="pd-measurer__btn pd-measurer__btn--primary" onClick={captureFromCamera}>
                                                        Capture Photo
                                                    </button>
                                                    <button className="pd-measurer__btn pd-measurer__btn--secondary" onClick={stopCamera}>
                                                        Cancel
                                                    </button>
                                                </>
                                            )}
                                        </div>
                                    </>
                                ) : (
                                    <div className="pd-measurer__preview">
                                        <img src={imagePreview} alt="Captured" className="pd-measurer__preview-image" />
                                        <button className="pd-measurer__preview-remove" onClick={removePreview} title="Remove image">
                                            <CloseIcon />
                                        </button>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Reference selector */}
                        <div className="pd-measurer__reference">
                            <label className="pd-measurer__reference-label">Reference Object (Optional)</label>
                            <select className="pd-measurer__reference-select" value={referenceType} onChange={(e) => setReferenceType(e.target.value as ReferenceType)}>
                                {referenceOptions.map((opt) => (
                                    <option key={opt.value} value={opt.value}>
                                        {opt.label}
                                    </option>
                                ))}
                            </select>
                            {referenceType !== "none" && <p className="pd-measurer__reference-hint">{referenceOptions.find((o) => o.value === referenceType)?.hint}</p>}
                        </div>

                        {/* Submit button */}
                        <button className="pd-measurer__btn pd-measurer__btn--primary" onClick={handleSubmit} disabled={!imageFile}>
                            Measure PD
                        </button>
                    </>
                )}

                {/* Processing step */}
                {step === "processing" && (
                    <div className="pd-measurer__loading">
                        <div className="pd-measurer__spinner" />
                        <p className="pd-measurer__loading-text">Analyzing your photo...</p>
                    </div>
                )}

                {/* Results step */}
                {step === "results" && result && (
                    <div className="pd-measurer__results">
                        <div className="pd-measurer__results-header">
                            <CheckIcon />
                            <h3 className="pd-measurer__results-title">Measurement Complete</h3>
                        </div>

                        {/* Confidence badge */}
                        <div className="pd-measurer__confidence">
                            <span className={`pd-measurer__confidence-dot pd-measurer__confidence-dot--${getConfidenceLevel(result.confidence_score).class}`} />
                            <span className="pd-measurer__confidence-text">
                                {getConfidenceLevel(result.confidence_score).label} Confidence ({(result.confidence_score * 100).toFixed(0)}%)
                            </span>
                        </div>

                        {/* Measurements grid */}
                        <div className="pd-measurer__measurements">
                            <div className="pd-measurer__measurement">
                                <div className="pd-measurer__measurement-label">Overall PD</div>
                                <div className="pd-measurer__measurement-value">
                                    {result.overall_pd_mm}
                                    <span className="pd-measurer__measurement-unit"> mm</span>
                                </div>
                            </div>
                            <div className="pd-measurer__measurement">
                                <div className="pd-measurer__measurement-label">Left Eye</div>
                                <div className="pd-measurer__measurement-value">
                                    {result.left_pd_mm}
                                    <span className="pd-measurer__measurement-unit"> mm</span>
                                </div>
                            </div>
                            <div className="pd-measurer__measurement">
                                <div className="pd-measurer__measurement-label">Right Eye</div>
                                <div className="pd-measurer__measurement-value">
                                    {result.right_pd_mm}
                                    <span className="pd-measurer__measurement-unit"> mm</span>
                                </div>
                            </div>
                        </div>

                        {/* Method info */}
                        <div className="pd-measurer__method">
                            <div className="pd-measurer__method-label">Measurement Method</div>
                            <div className="pd-measurer__method-value">
                                {result.method === "iris_estimation" ? "Iris Diameter Estimation" : "Reference Object Calibration"}
                            </div>
                            <div className="pd-measurer__method-label" style={{ marginTop: "8px" }}>
                                Accuracy
                            </div>
                            <div className="pd-measurer__method-value">
                                ±{result.error_margin.value_mm}mm (±{result.error_margin.percentage}%)
                            </div>
                        </div>

                        {/* Disclaimer */}
                        <div className="pd-measurer__disclaimer">
                            <div className="pd-measurer__disclaimer-title">
                                <WarningIcon />
                                Important Notice
                            </div>
                            <p className="pd-measurer__disclaimer-text">{result.disclaimer}</p>
                        </div>

                        {/* Actions */}
                        <div className="pd-measurer__actions">
                            <button className="pd-measurer__btn pd-measurer__btn--secondary" onClick={handleReset}>
                                Measure Again
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default PDMeasurer;
