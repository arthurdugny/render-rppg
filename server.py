#!/usr/bin/env python3
"""
Serveur rPPG « facepalm » — S5

Endpoints :
  WS   /ws             streaming frames (START | binary | END) → palm_g par frame
  GET  /               rppg_live.html
  POST /process_video  inférence CNN1D paume sur vidéo enregistrée
"""

import asyncio, json, queue, sys, tempfile, threading, time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
from metrics import extract_all_metrics                                    # type: ignore
from inference_paume_video import run_cnn1d_full, suppress_outlier_peaks   # type: ignore

# ── MediaPipe Hands ──────────────────────────────────────────────────
_hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)

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

# ── Extraction paume ─────────────────────────────────────────────────

PALM_ROIS = {
    "centre":      [0, 5, 9, 13, 17],
    "thenar":      [0, 1, 2, 5],
    "hypothenar":  [0, 13, 17],
    "base_doigts": [5, 9, 13, 17],
}
PALM_ROI_NAMES = list(PALM_ROIS.keys())
_ERODE5 = np.ones((5, 5), np.uint8)

def _skin_hsv(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 15, 40], dtype=np.uint8),   np.array([25, 200, 255], dtype=np.uint8))
    m2 = cv2.inRange(hsv, np.array([165, 15, 40], dtype=np.uint8), np.array([180, 200, 255], dtype=np.uint8))
    return cv2.morphologyEx(cv2.bitwise_or(m1, m2), cv2.MORPH_CLOSE, _ERODE5)

def _roi_mask_palm(shape, landmarks, indices):
    h, w = shape[:2]
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices], dtype=np.int32)
    hull = cv2.convexHull(pts)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    return cv2.erode(m, _ERODE5, iterations=1)

def _region_rgb_palm(frame_bgr, mask_roi, mask_skin, sigma=2.0):
    final = cv2.bitwise_and(mask_roi, mask_skin)
    pix = frame_bgr[final > 0].astype(np.float32)
    if len(pix) < 30:
        return np.nan, np.nan, np.nan
    keep = np.ones(len(pix), dtype=bool)
    for c in range(3):
        med, std = np.median(pix[:, c]), np.std(pix[:, c])
        if std > 0:
            keep &= np.abs(pix[:, c] - med) <= sigma * std
    pix = pix[keep]
    if len(pix) < 30:
        return np.nan, np.nan, np.nan
    return float(pix[:, 2].mean()), float(pix[:, 1].mean()), float(pix[:, 0].mean())

def extract_palm(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    skin = _skin_hsv(frame_bgr)
    out = {}
    for name, idx in PALM_ROIS.items():
        m = _roi_mask_palm(frame_bgr.shape, lm, idx)
        out[name] = _region_rgb_palm(frame_bgr, m, skin)
    return out

# ── Session ───────────────────────────────────────────────────────────

class Session:
    def __init__(self):
        self.frame_queue: queue.Queue = queue.Queue()
        self.palm_rows: list = []
        self.done = threading.Event()
        self.fps = 30.0
        self.cnn1d = None
        self.frame_count = 0
        self.last_palm_g: Optional[float] = None
        self._t0 = None
        self._t1 = None

    def start(self, cnn1d, fps):
        self.cnn1d = cnn1d
        self.fps = fps
        self.done.clear()
        self.palm_rows = []
        self.frame_count = 0
        self.last_palm_g = None
        self._t0 = self._t1 = None
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        print(f"[Session] Démarrée (fps={fps})")

    def push_frame(self, jpeg):
        self.frame_queue.put(jpeg)

    def end(self):
        self.frame_queue.put(None)

    def _loop(self):
        try:
            self._loop_inner()
        except Exception:
            import traceback
            traceback.print_exc()
            self.done.set()

    def _loop_inner(self):
        while True:
            item = self.frame_queue.get()
            if item is None:
                break
            t = time.monotonic()
            self._t0 = self._t0 or t
            self._t1 = t

            frame = cv2.imdecode(np.frombuffer(item, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            palm = extract_palm(frame)
            if palm:
                palm_row = []
                for name in PALM_ROI_NAMES:
                    palm_row.extend(palm[name])
                g_vals = [palm_row[i] for i in (1, 4, 7, 10) if not np.isnan(palm_row[i])]
                self.last_palm_g = float(np.mean(g_vals)) if g_vals else None
            else:
                palm_row = [np.nan] * 12
                self.last_palm_g = None
            self.palm_rows.append(palm_row)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                print(f"[Session] {self.frame_count} frames")

        if self._t0 and self._t1 and self.frame_count > 1:
            self.fps = (self.frame_count - 1) / (self._t1 - self._t0)

        self.done.set()


# ── FastAPI ───────────────────────────────────────────────────────────

app = FastAPI()
cnn1d_model: Optional[object] = None
active_session: Optional[Session] = None


@app.on_event("startup")
def startup():
    global cnn1d_model
    cnn1d_model = load_cnn1d()


@app.get("/")
def index():
    return FileResponse(str(HTML_PATH)) if HTML_PATH.exists() else HTMLResponse("<h1>rppg_live.html introuvable</h1>")


@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    """Lance inference_paume_video.py en subprocess — résultat identique à la commande manuelle."""
    import subprocess, json as _json
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    out_json = tmp_path + ".json"
    try:
        script = HERE / "inference_paume_video.py"
        print(f"[process_video] subprocess sur {tmp_path}…")
        proc = await asyncio.get_event_loop().run_in_executor(None, lambda: subprocess.run(
            [sys.executable, str(script),
             "--video", tmp_path,
             "--checkpoint", str(CKPT_PATH),
             "--out_json", out_json],
            capture_output=True, text=True, timeout=300
        ))
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
            return JSONResponse({"error": "inference_paume_video.py a échoué"}, status_code=500)
        result = _json.loads(Path(out_json).read_text())
        print(f"[process_video] HR = {result['hr_bpm']} bpm")
        return JSONResponse(result)
    except Exception:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": "Échec traitement vidéo"}, status_code=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(out_json).unlink(missing_ok=True)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global active_session
    await ws.accept()
    session = Session()
    active_session = session
    loop = asyncio.get_event_loop()
    try:
        while True:
            msg = await ws.receive()
            if "text" in msg:
                data = json.loads(msg["text"])
                if data.get("type") == "START":
                    session.start(cnn1d_model, float(data.get("fps", 30)))
                    await ws.send_text(json.dumps({"status": "started"}))
                elif data.get("type") == "END":
                    session.end()
                    await loop.run_in_executor(None, session.done.wait, 600)
                    await ws.send_text(json.dumps({
                        "status":  "done",
                        "frames":  session.frame_count,
                    }))
            elif "bytes" in msg:
                session.push_frame(msg["bytes"])
                await ws.send_text(json.dumps({
                    "status": "processing",
                    "queued": session.frame_queue.qsize(),
                    "done":   session.frame_count,
                    "palm_g": session.last_palm_g,
                }))
    except WebSocketDisconnect:
        print("[WS] Client déconnecté")
    except Exception as e:
        print(f"[WS] Erreur : {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 8000)), reload=False)
