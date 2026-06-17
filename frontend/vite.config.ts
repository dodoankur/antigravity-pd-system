import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import { viteStaticCopy } from "vite-plugin-static-copy";
import type { Connect } from "vite";
import fs from "fs";

// ── Custom router plugin ──────────────────────────────────────────────────────
// Serves iframe-test.html at /  (new home)
// Serves old-version.html   at /old-version
function customRouterPlugin() {
    return {
        name: "custom-router",
        configureServer(server: any) {
            server.middlewares.use(
                (req: Connect.IncomingMessage, res: any, next: Connect.NextFunction) => {
                    const url = (req.url ?? "").split("?")[0];

                    if (url === "/" || url === "/index.html") {
                        const html = fs.readFileSync(resolve(__dirname, "iframe-test.html"), "utf-8");
                        server.transformIndexHtml(url, html).then((t: string) => {
                            res.setHeader("Content-Type", "text/html");
                            res.end(t);
                        });
                        return;
                    }

                    if (url === "/old-version" || url === "/old-version/" || url === "/old-version.html") {
                        const html = fs.readFileSync(resolve(__dirname, "old-version.html"), "utf-8");
                        server.transformIndexHtml(url, html).then((t: string) => {
                            res.setHeader("Content-Type", "text/html");
                            res.end(t);
                        });
                        return;
                    }

                    next();
                }
            );
        },
    };
}

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
                {
                    src: "node_modules/@mediapipe/camera_utils/*.js",
                    dest: "mediapipe/camera_utils",
                },
            ],
        }),
        customRouterPlugin(),
    ],
    // Multi-page build: iframe-test (home) + old-version + embed
    build: {
        rollupOptions: {
            input: {
                main:            resolve(__dirname, "iframe-test.html"),
                "old-version":   resolve(__dirname, "old-version.html"),
                embed:           resolve(__dirname, "embed.html"),
            },
        },
    },
    resolve: {
        alias: {
            "@": resolve(__dirname, "./src"),
        },
    },
    optimizeDeps: {
        exclude: ["@mediapipe/face_mesh", "@mediapipe/camera_utils"],
    },
    define: {
        "import.meta.env.VITE_API_BASE_URL": JSON.stringify(process.env.VITE_API_BASE_URL || ""),
    },
    appType: "mpa",
    server: {
        allowedHosts: [".trycloudflare.com"],
        proxy: {
            "/api": {
                target: "http://localhost:8000",
                changeOrigin: true,
            },
        },
    },
});
