import { defineConfig } from "vite";
import path from "node:path";

export default defineConfig({
  base: "/static/app/",
  build: {
    outDir: path.resolve(__dirname, "../static/app"),
    emptyOutDir: true,
  },
});
