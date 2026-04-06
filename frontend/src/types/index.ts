/**
 * Type definitions for PD Measurement API
 */

export type ReferenceType =
    | "none"
    | "credit_card"
    | "coin_gbp_1p"
    | "coin_gbp_2p"
    | "coin_gbp_5p"
    | "coin_gbp_10p"
    | "coin_gbp_20p"
    | "coin_gbp_50p"
    | "coin_gbp_1"
    | "coin_gbp_2"
    | "ruler";

export type MeasurementMethod = "iris_estimation" | "reference_object";

export interface ErrorMargin {
    value_mm: number;
    percentage: number;
    confidence_score: number;
}

export interface PDMeasurementResult {
    overall_pd_mm: number;
    left_pd_mm: number;
    right_pd_mm: number;
    method: MeasurementMethod;
    model_used: string;
    confidence_score: number;
    error_margin: ErrorMargin;
    disclaimer: string;
    face_detected: boolean;
    eyes_detected: boolean;
    reference_detected: boolean | null;
    reference_scale_factor: number | null;
}

export interface ReferenceTypeOption {
    value: ReferenceType;
    label: string;
    description: string;
    accuracy: string;
}

export interface PDMeasurerProps {
    /** API endpoint for PD measurement (default: /api/pd/measure) */
    apiEndpoint?: string;
    /** Callback when measurement is complete */
    onMeasurement?: (result: PDMeasurementResult) => void;
    /** Callback on error */
    onError?: (error: string) => void;
    /** Custom class name for styling */
    className?: string;
    /** Primary color theme */
    primaryColor?: string;
    /** Base path for MediaPipe Face Mesh files (default: /mediapipe/face_mesh) */
    mediapipeBasePath?: string;
}
