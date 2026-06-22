import React from "react";
import ReactDOM from "react-dom/client";
import { IframeUI } from "./components/PDMeasurer/IframeUI";
import { PDMeasurementResult } from "./types";

/**
 * Embedded entry point — rendered at /embed.
 *
 * Mode detection (in priority order):
 *   1. Popup / Tab  — window.opener !== null
 *      → postMessage to opener + BroadcastChannel + auto-close after 1.5 s
 *   2. Iframe       — window.parent !== window
 *      → postMessage to parent (existing behaviour, no regression)
 *   3. Standalone   — direct navigation
 *      → no postMessage
 */

const isPopup  = window.opener !== null;
const isIframe = !isPopup && window.parent !== window;

function handleMeasurement(result: PDMeasurementResult) {
    console.log("[embed] PD result:", result);

    const payload = {
        type:               "PD_RESULT",
        overall_pd_mm:      result.overall_pd_mm,
        right_pd_mm:        result.right_pd_mm,
        left_pd_mm:         result.left_pd_mm,
        confidence_score:   result.confidence_score,
        age_group_used:     result.age_group_used,
        best_frame_dataurl: result.best_frame_dataurl ?? null,
    };

    if (isPopup) {
        // Tab / popup mode — send result to opener page
        try {
            window.opener.postMessage(payload, "*");
        } catch {
            // opener may have navigated away — ignore
        }

        // BroadcastChannel as reliable same-origin fallback
        try {
            const bc = new BroadcastChannel("pd_result");
            bc.postMessage(payload);
            bc.close();
        } catch {
            // BroadcastChannel not supported (very old Safari) — opener postMessage is enough
        }

        // Auto-close the tab after a short delay so user sees the result screen
        setTimeout(() => window.close(), 1500);

    } else if (isIframe) {
        // Iframe mode — existing behaviour unchanged
        window.parent.postMessage(payload, "*");
    }
    // Standalone: no postMessage needed
}

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <IframeUI onMeasurement={handleMeasurement} />
    </React.StrictMode>,
);
