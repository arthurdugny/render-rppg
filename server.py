#!/usr/bin/env python3
"""
Serveur rPPG — S6 (Render-compatible, CNN1D paume uniquement)

Endpoints :
  WS  /ws   START → END{palm_rows, fps} → {status:done, cnn1d:{...}}
  GET /      rppg_live.html
"""

import asyncio, json, sys
from pathlib import Path

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

# ── Imports CNN1D ─────────────────────────────────────────────────────
sys.path.insert(0, str(HERE))
from cnn1d_model import CNN1D_rPPG  # type: ignore

# ── CNN1D ─────────────────────────────────────────────────────────────

def load_cnn1d():
    model = CNN1D_rPPG(in_channels=12).to(_device)
    if CKPT_PATH.exists():
        ckpt = torch.load(str(CKPT_PATH), map_location=_device)
        model.load_state_dict(ckpt["model"])
        print(f"[CNN1D] Checkpoint chargé (époque {ckpt.get('epoch', '?')})")
    else:
        print(f"[CNN1D] ATTENTION : {CKPT_PATH} introuvable")
    return model.eval()


def _suppress_outliers(signal: np.ndarray, k: float = 3.5) -> np.ndarray:
    med = np.median(signal)
    mad = np.median(np.abs(signal - med))
    if mad < 1e-8:
        return signal
    return np.clip(signal, med - k * mad, med + k * mad)


def _run_cnn1d_overlap(arr: np.ndarray, fps: float, model, clip_len: int = 512) -> np.ndarray:
    T = arr.shape[0]
    arr = arr.copy().astype(np.float32)
    for c in range(arr.shape[1]):
        col = arr[:, c]
        nan = np.isnan(col)
        if nan.any() and not nan.all():
            ok = np.where(~nan)[0]
            col[nan] = np.interp(np.where(nan)[0], ok, col[ok])
        elif nan.all():
            col[:] = 0.0
    mu  = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-6
    arr = (arr - mu) / std

    rppg_full = np.zeros(T, np.float64)
    weight    = np.zeros(T, np.float64)
    win       = np.hanning(clip_len)
    stride    = clip_len // 2
    starts    = list(range(0, max(1, T - clip_len + 1), stride))
    if T >= clip_len and (T - clip_len) not in starts:
        starts.append(T - clip_len)

    model.eval()
    with torch.no_grad():
        for s in starts:
            e = min(s + clip_len, T)
            clip = arr[s:e]
            if len(clip) < clip_len:
                clip = np.concatenate([clip, np.zeros((clip_len - len(clip), 12), np.float32)])
            x = torch.from_numpy(clip.T[np.newaxis]).to(_device)
            rppg, _ = model(x)
            sig = rppg.squeeze(0).cpu().numpy()
            sig_std = sig.std()
            if sig_std > 1e-6:
                sig = (sig - sig.mean()) / sig_std
            L = e - s
            rppg_full[s:e] += sig[:L] * win[:L]
            weight[s:e]    += win[:L]

    mask = weight > 0
    rppg_full[mask] /= weight[mask]
    return rppg_full


def _local_maxima(a: np.ndarray, min_val: float) -> np.ndarray:
    """Indices des maxima locaux stricts au-dessus de min_val — implémentation
    numpy pure (pas de dépendance scipy, cf. contrainte mémoire Render)."""
    if len(a) < 3:
        return np.array([], dtype=int)
    is_max = (a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]) & (a[1:-1] >= min_val)
    return np.where(is_max)[0] + 1


def _pick_hr_harmonic(freqs_bpm: np.ndarray, spec: np.ndarray, band=(45.0, 180.0),
                       harmonic_tol_bpm: float = 6.0,
                       harmonic_weights=(0.7, 0.45)) -> tuple:
    """Sélectionne le pic HR dans la bande cardiaque par sommation
    harmonique (fondamental + 2e/3e harmonique pondérés, cf. subharmonic
    summation) plutôt qu'un simple argmax — un artefact de dérive/mouvement
    résiduel n'a normalement pas d'harmoniques cohérentes avec lui-même,
    contrairement à un vrai pouls. Le bruit de fond contribue de façon
    similaire au score de chaque candidat (il s'additionne aux deux), donc
    ce qui fait la différence est bien l'énergie réelle aux harmoniques,
    pas seulement l'amplitude brute du fondamental."""
    band_mask = (freqs_bpm >= band[0]) & (freqs_bpm <= band[1])
    freqs_b, spec_b = freqs_bpm[band_mask], spec[band_mask]
    if len(spec_b) == 0:
        return float("nan"), 0.0

    idx_peaks = _local_maxima(spec_b, (np.median(spec_b) + 1e-9) * 1.15)
    if len(idx_peaks) == 0:
        idx_peaks = np.array([int(np.argmax(spec_b))])

    def spec_near(f_bpm: float) -> float:
        hmask = np.abs(freqs_bpm - f_bpm) <= harmonic_tol_bpm
        return float(spec[hmask].max()) if hmask.any() else 0.0

    def comb_score(f0: float) -> float:
        s = spec_near(f0)
        for k, w in zip((2, 3), harmonic_weights):
            if k * f0 > freqs_bpm[-1]:
                break
            s += w * spec_near(k * f0)
        return s

    scores = [comb_score(freqs_b[i]) for i in idx_peaks]
    best = idx_peaks[int(np.argmax(scores))]
    confidence = float(spec_b[best] / (spec_b.max() + 1e-9))
    return float(freqs_b[best]), confidence


def run_cnn1d(palm_rows, fps, model):
    arr = np.array(palm_rows, dtype=np.float32)
    if arr.shape[0] < 30 or model is None:
        return {"error": "Pas assez de frames paume"}
    try:
        rppg = _suppress_outliers(_run_cnn1d_overlap(arr, fps, model))
        N = len(rppg)
        # Passe-haut par soustraction de moyenne glissante (élimine toute dérive < ~0.6 Hz)
        win_hp = max(3, int(round(fps * 1.6)))  # fenêtre ~1.6 s → coupe < ~0.6 Hz (36 bpm)
        if win_hp % 2 == 0: win_hp += 1
        ma = np.convolve(rppg, np.ones(win_hp) / win_hp, mode='same')
        rppg_d = rppg - ma
        freqs_bpm = np.fft.rfftfreq(N, d=1.0 / fps) * 60  # Hz → bpm
        spec      = np.abs(np.fft.rfft(rppg_d * np.hanning(N)))
        mask      = (freqs_bpm >= 45) & (freqs_bpm <= 180)
        freqs_m, spec_m = freqs_bpm[mask], spec[mask]
        hr_raw, _hr_conf = _pick_hr_harmonic(freqs_bpm, spec)
        hr = round(hr_raw, 1) if len(freqs_m) > 0 and not np.isnan(hr_raw) else float("nan")
        step = max(1, N // 600)
        print(f"[CNN1D] HR = {hr} bpm ({N} frames, {fps:.2f} fps)")
        return {
            "rppg_signal": rppg[::step].tolist(),
            "freqs_bpm":   freqs_m.tolist(),
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
    except Exception:
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
