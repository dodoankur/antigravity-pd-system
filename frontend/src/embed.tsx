import React from "react";
import ReactDOM from "react-dom/client";
import { PDMeasurer } from "./components/PDMeasurer";
import componentStyles from "./components/PDMeasurer/styles.css?inline";
import type { PDMeasurementResult } from "./types";

/**
 * Embeddable Widget Entry Point
 *
 * This file is the entry point for the embeddable widget build.
 * It creates a global PDMeasurementWidget object that can be used to:
 * 1. Initialize the widget in a container element
 * 2. Configure the widget with custom options
 *
 * Usage in vanilla HTML:
 * <script src="pd-widget.js"></script>
 * <div id="pd-widget-container"></div>
 * <script>
 *   PDMeasurementWidget.init('#pd-widget-container', {
 *     apiEndpoint: 'https://your-api.com/api/pd/measure',
 *     onMeasurement: function(result) { console.log(result); }
 *   });
 * </script>
 */

interface WidgetOptions {
    apiEndpoint?: string;
    /** Typed callback — receives the full PDMeasurementResult from the API */
    onMeasurement?: (result: PDMeasurementResult) => void;
    onError?: (error: string) => void;
    primaryColor?: string;
}

interface WidgetInstance {
    root: ReactDOM.Root;
    container: HTMLElement;
    shadowRoot: ShadowRoot;
}

// Store widget instances
const instances = new Map<string, WidgetInstance>();

/**
 * Create and inject widget styles into shadow DOM
 */
function injectStyles(shadowRoot: ShadowRoot): void {
    // Get all styles from the bundle
    const styleElement = document.createElement("style");
    styleElement.textContent = `
    /* CSS Variables for theming */
    :host {
      --pd-primary: #4f46e5;
      --pd-primary-hover: #4338ca;
      --pd-secondary: #6b7280;
      --pd-success: #10b981;
      --pd-warning: #f59e0b;
      --pd-error: #ef4444;
      --pd-bg: #ffffff;
      --pd-bg-secondary: #f9fafb;
      --pd-border: #e5e7eb;
      --pd-text: #111827;
      --pd-text-muted: #6b7280;
      --pd-radius: 12px;
      --pd-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
      --pd-shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
      
      display: block;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    }
    
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
  `;

    // Import the compiled styles
    const compiledStyles = document.createElement("style");
    compiledStyles.textContent = componentStyles;

    shadowRoot.appendChild(styleElement);
    shadowRoot.appendChild(compiledStyles);
}

/**
 * Initialize the PD Measurement Widget
 */
function init(selector: string | HTMLElement, options: WidgetOptions = {}): void {
    const container = typeof selector === "string" ? (document.querySelector(selector) as HTMLElement) : selector;

    if (!container) {
        console.error("[PDMeasurementWidget] Container not found:", selector);
        return;
    }

    const containerId = container.id || `pd-widget-${Date.now()}`;
    container.id = containerId;

    // Clean up existing instance
    if (instances.has(containerId)) {
        destroy(containerId);
    }

    // Create shadow DOM for style isolation
    const shadowHost = document.createElement("div");
    shadowHost.className = "pd-widget-shadow-host";
    container.appendChild(shadowHost);

    const shadowRoot = shadowHost.attachShadow({ mode: "open" });
    injectStyles(shadowRoot);

    // Create mount point inside shadow DOM
    const mountPoint = document.createElement("div");
    mountPoint.className = "pd-widget-mount";
    shadowRoot.appendChild(mountPoint);

    // Create React root and render
    const root = ReactDOM.createRoot(mountPoint);
    root.render(
        React.createElement(PDMeasurer, {
            apiEndpoint: options.apiEndpoint,
            onMeasurement: options.onMeasurement,
            onError: options.onError,
            primaryColor: options.primaryColor,
        }),
    );

    // Store instance
    instances.set(containerId, {
        root,
        container,
        shadowRoot,
    });

    console.log("[PDMeasurementWidget] Initialized in:", containerId);
}

/**
 * Destroy a widget instance
 */
function destroy(containerId: string): void {
    const instance = instances.get(containerId);
    if (instance) {
        instance.root.unmount();
        instance.container.innerHTML = "";
        instances.delete(containerId);
        console.log("[PDMeasurementWidget] Destroyed:", containerId);
    }
}

/**
 * Get all active widget instances
 */
function getInstances(): string[] {
    return Array.from(instances.keys());
}

// Expose global API
const PDMeasurementWidget = {
    init,
    destroy,
    getInstances,
    version: "1.0.0",
};

// Make available globally
(window as unknown as { PDMeasurementWidget: typeof PDMeasurementWidget }).PDMeasurementWidget = PDMeasurementWidget;

export default PDMeasurementWidget;
