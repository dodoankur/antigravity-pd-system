import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

/**
 * Vite configuration for building the embeddable widget
 * Outputs a single IIFE bundle that can be included via script tag
 */
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            "@": resolve(__dirname, "./src"),
        },
    },
    define: {
        "process.env.NODE_ENV": JSON.stringify("production"),
    },
    build: {
        lib: {
            entry: resolve(__dirname, "src/embed.tsx"),
            name: "PDMeasurementWidget",
            fileName: () => "pd-widget.js",
            formats: ["iife"],
        },
        rollupOptions: {
            // Don't externalize React - bundle it
            external: [],
            output: {
                // Global variables for external dependencies
                globals: {},
                // Ensure styles are injected
                inlineDynamicImports: true,
            },
        },
        cssCodeSplit: false,
        minify: "esbuild",
        outDir: "dist/widget",
    },
});
