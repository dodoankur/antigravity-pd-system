import React from "react";
import ReactDOM from "react-dom/client";
import { PDMeasurer } from "./components/PDMeasurer";
import "./index.css";

/**
 * Development entry point
 * Renders the PD Measurer widget in a demo container
 */
ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <div className="demo-container">
            <h1>PD Measurement Tool</h1>
            <p className="demo-subtitle">Measure your pupil distance for accurate eyewear fitting</p>
            <PDMeasurer
                onMeasurement={(result) => {
                    console.log("Measurement result:", result);
                }}
            />
        </div>
    </React.StrictMode>,
);
