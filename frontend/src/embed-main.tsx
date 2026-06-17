import React from "react";
import ReactDOM from "react-dom/client";
import { IframeUI } from "./components/PDMeasurer/IframeUI";
import { PDMeasurementResult } from "./types";

/**
 * Embedded entry point — rendered at /embed
 * Loads the Lenskart-inspired full-screen IframeUI and
 * posts the result to the parent POS window via postMessage.
 */
function handleMeasurement(result: PDMeasurementResult) {
    console.log("[embed] PD result:", result);
    // Always post to parent — the POS portal listens for this
    window.parent.postMessage(
        {
            type:          "PD_RESULT",
            overall_pd_mm: result.overall_pd_mm,
            right_pd_mm:   result.right_pd_mm,
            left_pd_mm:    result.left_pd_mm,
        },
        "*",
    );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <IframeUI onMeasurement={handleMeasurement} />
    </React.StrictMode>,
);
