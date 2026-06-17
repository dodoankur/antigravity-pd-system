import React from "react";
import ReactDOM from "react-dom/client";
import { PDMeasurer } from "./components/PDMeasurer";
import "./index.css";

/** True when the widget is embedded inside an iframe (e.g. POS portal) */
const isEmbedded = window.self !== window.top;

/** Shared measurement handler — sends result to parent frame when embedded */
function handleMeasurement(result: { overall_pd_mm: number; right_pd_mm: number; left_pd_mm: number }) {
    console.log("Measurement result:", result);
    if (isEmbedded) {
        window.parent.postMessage(
            {
                type: "PD_RESULT",
                overall_pd_mm: result.overall_pd_mm,
                right_pd_mm:   result.right_pd_mm,
                left_pd_mm:    result.left_pd_mm,
            },
            "*", // parent is on a different origin (POS portal)
        );
    }
}

// When embedded in an iframe, remove the gradient background from <body>
// so only the PDMeasurer card is visible — no demo chrome.
if (isEmbedded) {
    document.body.style.background = "transparent";
    document.body.style.minHeight  = "unset";
}

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        {isEmbedded ? (
            // ── Embedded mode: bare component only, no demo wrapper ──
            <PDMeasurer onMeasurement={handleMeasurement} />
        ) : (
            // ── Standalone / dev mode: full demo page with title ──
            <div className="demo-container">
                <h1>PD Measurement Tool</h1>
                <p className="demo-subtitle">Measure your pupil distance for accurate eyewear fitting</p>
                <PDMeasurer onMeasurement={handleMeasurement} />
            </div>
        )}
    </React.StrictMode>,
);
