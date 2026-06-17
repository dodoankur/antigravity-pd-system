import React from "react";
import ReactDOM from "react-dom/client";
import { IframeUI } from "./components/PDMeasurer/IframeUI";
import { PDMeasurementResult } from "./types";

/**
 * Embedded entry point — rendered at /embed
 * Posts PD result + best frame image (base64) to the parent POS window via postMessage.
 */
function handleMeasurement(result: PDMeasurementResult) {
    console.log("[embed] PD result:", result);
    window.parent.postMessage(
        {
            type: "PD_RESULT",
            overall_pd_mm:      result.overall_pd_mm,
            right_pd_mm:        result.right_pd_mm,
            left_pd_mm:         result.left_pd_mm,
            confidence_score:   result.confidence_score,
            age_group_used:     result.age_group_used,
            best_frame_dataurl: result.best_frame_dataurl ?? null,
        },
        "*",
    );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <IframeUI onMeasurement={handleMeasurement} />
    </React.StrictMode>,
);
