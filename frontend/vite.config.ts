import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Two builds from one code base (docs/decisions.md D38):
//   npm run dev / build        -> talks to the FastAPI backend (/api, proxied in dev)
//   npm run build:static       -> reads pre-exported files from public/demo/, for GitHub Pages
// The static build is served from https://<user>.github.io/cardiac-mri-analyzer/, hence `base`.
export default defineConfig(({ mode }) => ({
  base: mode === "static" ? "/cardiac-mri-analyzer/" : "/",
  plugins: [react(), tailwindcss()],
  define: {
    "import.meta.env.VITE_DATA_MODE": JSON.stringify(mode === "static" ? "static" : "api"),
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  test: {
    environment: "node",
  },
}));
