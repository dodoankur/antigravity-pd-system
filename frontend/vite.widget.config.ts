import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import { viteStaticCopy } from "vite-plugin-static-copy";

/**
 * Vite configuration for building the embeddable widget
 * Outputs a single IIFE bundle that can be included via script tag
 */
export default defineConfig({
    plugins: [
        react(),
        viteStaticCopy({
            targets: [
                {
                    src: "node_modules/@mediapipe/face_mesh/*.{js,wasm,data,binarypb}",
                    dest: "mediapipe/face_mesh",
                },
                {
                    src: "node_modules/@mediapipe/camera_utils/*.js",
                    dest: "mediapipe/camera_utils",
                },
            ],
        }),
    ],
    resolve: {
        alias: {
            "@": resolve(__dirname, "./src"),
        },
    },
    optimizeDeps: {
        exclude: ["@mediapipe/face_mesh", "@mediapipe/camera_utils"],
    },
    define: {
        "process.env.NODE_ENV": JSON.stringify("production"),
        "import.meta.env.VITE_API_BASE_URL": JSON.stringify(process.env.VITE_API_BASE_URL || ""),
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
