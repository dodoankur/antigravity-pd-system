import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import { viteStaticCopy } from "vite-plugin-static-copy";

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [
        react(),
        viteStaticCopy({
            targets: [
                {
                    src: "node_modules/@mediapipe/face_mesh/*.{js,wasm,data,binarypb}",
                    dest: "mediapipe/face_mesh",
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
    server: {
        port: 3000,
        allowedHosts: [".trycloudflare.com"],
        proxy: {
            "/api": {
                target: "http://localhost:8000",
                changeOrigin: true,
            },
        },
    },
});
