#!/usr/bin/env python3
"""
Serveur rPPG « facepalm » — S5

Endpoints :
  WS  /ws   START → END{palm_rows, fps} → {status:done, cnn1d:{...}}
  GET /      rppg_live.html
"""

import asyncio, json, sys, threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

# ── Device ───────────────────────────────────────────────────────────
_device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)
print(f"[Device] {_device}")

# ── Chemins ──────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
CKPT_PATH = HERE / "best_cnn1d.pth"
HTML_PATH = HERE / "rppg_live.html"

# ── Imports pipeline ─────────────────────────────────────────────────
sys.path.insert(0, str(HERE))
from metrics import extract_all_metrics                                  # type: ignore
from inference_paume_video import run_cnn1d_full, suppress_outlier_peaks # type: ignore

# ── CNN1D ────────────────────────────────────────────────────────────

def load_cnn1d():
    from cnn1d_model import CNN1D_rPPG  # type: ignore
    model = CNN1D_rPPG(in_channels=12).to(_device)
    if CKPT_PATH.exists():
        ckpt = torch.load(str(CKPT_PATH), map_location=_device)
        model.load_state_dict(ckpt["model"])
        print(f"[CNN1D] Checkpoint S5 chargé (époque {ckpt.get('epoch', '?')})")
    else:
        print(f"[CNN1D] ATTENTION : {CKPT_PATH} introuvable")
    return model.eval()

# ── Inférence CNN1D ───────────────────────────────────────────────────

def run_cnn1d(palm_rows, fps, model):
    arr = np.array(palm_rows, dtype=np.float32)
    if arr.shape[0] < 30 or model is None:
        return {"error": "Pas assez de frames paume"}
    try:
        rppg = suppress_outlier_peaks(run_cnn1d_full(arr, model, fps, _device))
        m = extract_all_metrics(rppg, fps)
        N = len(rppg)
        freqs = np.fft.rfftfreq(N, d=1.0 / fps)
        spec  = np.abs(np.fft.rfft(rppg))
        mask  = (freqs >= 0.7) & (freqs <= 3.0)
        freqs_m, spec_m = freqs[mask], spec[mask]
        step = max(1, N // 600)
        hr = round(float(m.get("hr_bpm", float("nan"))), 1)
        print(f"[CNN1D] HR = {hr} bpm ({N} frames, {fps:.2f} fps)")
        return {
            "rppg_signal": rppg[::step].tolist(),
            "freqs_bpm":   (freqs_m * 60).tolist(),
            "spectrum":    (spec_m / (spec_m.max() + 1e-9)).tolist(),
            "hr_bpm":      hr,
            "n_frames":    N,
            "fps":         round(fps, 3),
        }
    except Exception:
        import traceback; traceback.print_exc()
        return {"error": "Échec CNN1D"}

# ── FastAPI ───────────────────────────────────────────────────────────

app = FastAPI()

class COOPCOEPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        return response

app.add_middleware(COOPCOEPMiddleware)

cnn1d_model = None

@app.on_event("startup")
def startup():
    global cnn1d_model
    cnn1d_model = load_cnn1d()

@app.get("/")
def index():
    return FileResponse(str(HTML_PATH)) if HTML_PATH.exists() else HTMLResponse("<h1>rppg_live.html introuvable</h1>")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            msg = await ws.receive()
            if "text" not in msg:
                continue
            data = json.loads(msg["text"])

            if data.get("type") == "START":
                await ws.send_text(json.dumps({"status": "started"}))

            elif data.get("type") == "END":
                palm_rows = data.get("palm_rows", [])
                fps = float(data.get("fps", 30))
                print(f"[WS] END reçu — {len(palm_rows)} frames, fps={fps:.2f}")
                result = await loop.run_in_executor(
                    None, run_cnn1d, palm_rows, fps, cnn1d_model
                )
                await ws.send_text(json.dumps({"status": "done", "cnn1d": result}))

    except WebSocketDisconnect:
        print("[WS] Client déconnecté")
    except Exception as e:
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
