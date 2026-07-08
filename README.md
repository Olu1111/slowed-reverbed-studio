# Slowed & Reverbed Studio

## Run it

```bash
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173).

## Notes

- Use Chrome or Edge — the video export feature (MediaRecorder + captureStream)
  is built against those; Safari/Firefox support is inconsistent.
- Must run over http://localhost or https:// — the Web Audio API won't work
  opened directly as a file:// URL.
