#!/usr/bin/env python3
"""
Serveur rPPG « facepalm » — S5

Pipeline :
  • FaceMesh + BiSeNet (optionnel) → BPM live CHROM/POS (face)
  • MediaPipe Hands + masque HSV  → CNN1D S5 → HR finale (paume)

Endpoints :
  WS  /ws            streaming frames  (START | binary | END)
  GET /              rppg_live.html
  GET /analyse       CHROM/POS sur le visage (après capture)
  GET /analyse_palm  CNN1D résultats (rPPG signal + spectre + HR)

Lancement :
  python server.py
"""

import asyncio, json, queue, sys, tempfile, threading, time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
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
HERE         = Path(__file__).parent
CKPT_PATH    = HERE / "best_cnn1d.pth"
HTML_PATH    = HERE / "rppg_live.html"
OUT_CSV      = HERE / "rppg_rgb.csv"
OUT_META     = HERE / "rppg_rgb_meta.json"

# ── Imports depuis les fichiers de référence semaine 5 ───────────────
sys.path.insert(0, str(HERE))
from metrics import extract_all_metrics                                    # type: ignore
from inference_paume_video import run_cnn1d_full, suppress_outlier_peaks   # type: ignore
from extract_rgb_paume_multi import extract_rgb_paume_multi                # type: ignore

# ── FaceMesh ─────────────────────────────────────────────────────────
_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)

# ── MediaPipe Hands ──────────────────────────────────────────────────
_hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)

def face_parsing_mask(image_bgr):
    return np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255

# ── CNN1D ────────────────────────────────────────────────────────────

def load_cnn1d():
    sys.path.insert(0, str(HERE))
    from cnn1d_model import CNN1D_rPPG  # type: ignore
    model = CNN1D_rPPG(in_channels=12).to(_device)
    if CKPT_PATH.exists():
        ckpt = torch.load(str(CKPT_PATH), map_location=_device)
        model.load_state_dict(ckpt["model"])
        print(f"[CNN1D] Checkpoint S5 chargé (époque {ckpt.get('epoch', '?')}, Pearson {ckpt.get('val_pearson', '?'):.3f})")
    else:
        print(f"[CNN1D] ATTENTION : {CKPT_PATH} introuvable")
    return model.eval()

# ── Extraction face ───────────────────────────────────────────────────

def polygon_mask(shape, landmarks, indices):
    h, w = shape[:2]
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask

_ERODE5 = np.ones((5, 5), np.uint8)

FACE_ROIS = {
    "front":            [103, 67, 109, 10, 338, 297, 332, 333, 168, 104],
    "joue_gauche":      [187, 214, 211, 57, 216, 203, 101, 118, 117, 123],
    "joue_droite":      [411, 434, 431, 287, 436, 423, 330, 347, 346, 352],
    "nez":              [1, 45, 134, 174, 197, 399, 363, 275],
    "sous_oeil_gauche": [24, 23, 22, 121, 47, 100, 119, 228],
    "sous_oeil_droit":  [252, 253, 254, 448, 348, 329, 277, 350],
}
FACE_ROI_NAMES = list(FACE_ROIS.keys())
_CHEEK = {"joue_gauche", "joue_droite"}

def _region_rgb(frame_bgr, mask, parsing, sigma=2.0):
    final = cv2.bitwise_and(mask, parsing)
    pix = frame_bgr[final > 0].astype(np.float32)
    if len(pix) < 50:
        return np.nan, np.nan, np.nan
    keep = np.ones(len(pix), dtype=bool)
    for c in range(3):
        med, std = np.median(pix[:, c]), np.std(pix[:, c])
        if std > 0:
            keep &= np.abs(pix[:, c] - med) <= sigma * std
    pix = pix[keep]
    if len(pix) < 50:
        return np.nan, np.nan, np.nan
    return float(pix[:, 2].mean()), float(pix[:, 1].mean()), float(pix[:, 0].mean())

def extract_face(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    parsing = face_parsing_mask(frame_bgr)
    out = {}
    for name, idx in FACE_ROIS.items():
        m = polygon_mask(frame_bgr.shape, lm, idx)
        if name in _CHEEK:
            m = cv2.erode(m, _ERODE5, iterations=1)
        out[name] = _region_rgb(frame_bgr, m, parsing)
    return out

# ── Extraction paume ─────────────────────────────────────────────────

PALM_ROIS = {
    "centre":      [0, 5, 9, 13, 17],
    "thenar":      [0, 1, 2, 5],
    "hypothenar":  [0, 13, 17],
    "base_doigts": [5, 9, 13, 17],
}
PALM_ROI_NAMES = list(PALM_ROIS.keys())

def _skin_hsv(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 15, 40], dtype=np.uint8),  np.array([25, 200, 255], dtype=np.uint8))
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

# ── CNN1D inférence ───────────────────────────────────────────────────

def cnn1d_inference(palm_rows, fps, model):
    arr = np.array(palm_rows, dtype=np.float32)
    if arr.shape[0] < 30:
        return None
    rppg = suppress_outlier_peaks(run_cnn1d_full(arr, model, fps, _device))
    N = len(rppg)

    m = extract_all_metrics(rppg, fps)
    hr_bpm = float(m.get("hr_bpm", float("nan")))

    # Spectre pour affichage
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    spec  = np.abs(np.fft.rfft(rppg))
    mask  = (freqs >= 0.7) & (freqs <= 3.0)
    freqs_m, spec_m = freqs[mask], spec[mask]
    spec_norm = (spec_m / (spec_m.max() + 1e-9)).tolist()

    step = max(1, N // 600)
    return {
        "rppg_signal": rppg[::step].tolist(),
        "freqs_bpm":   (freqs_m * 60).tolist(),
        "spectrum":    spec_norm,
        "hr_bpm":      round(hr_bpm, 1),
        "n_frames":    N,
        "fps":         round(fps, 3),
    }

# ── Session ───────────────────────────────────────────────────────────

class Session:
    def __init__(self):
        self.frame_queue: queue.Queue = queue.Queue()
        self.face_rows: list = []
        self.palm_rows: list = []
        self.done = threading.Event()
        self.fps = 30.0
        self.cnn1d   = None
        self.frame_count = 0
        self.cnn1d_result = None
        self._t0 = None
        self._t1 = None

    def start(self, cnn1d, fps):
        self.cnn1d   = cnn1d
        self.fps = fps
        self.done.clear()
        self.face_rows = []
        self.palm_rows = []
        self.frame_count = 0
        self.cnn1d_result = None
        self.last_palm_g: Optional[float] = None
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        print(f"[Session] Démarrée (fps={fps})")

    def push_frame(self, jpeg):
        self.frame_queue.put(jpeg)

    def end(self):
        self.frame_queue.put(None)

    def _loop(self):
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

            # Face
            face = extract_face(frame)
            row = {"frame": self.frame_count}
            for name in FACE_ROI_NAMES:
                if face:
                    row[f"{name}_r"], row[f"{name}_g"], row[f"{name}_b"] = face[name]
                else:
                    row[f"{name}_r"] = row[f"{name}_g"] = row[f"{name}_b"] = np.nan
            self.face_rows.append(row)

            # Palm
            palm = extract_palm(frame)
            if palm:
                palm_row = []
                for name in PALM_ROI_NAMES:
                    palm_row.extend(palm[name])
                # vert moyen sur les 4 ROI (indice 1, 4, 7, 10 dans palm_row = R G B R G B ...)
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

        self._export_face()
        self.done.set()

    def _export_face(self):
        df = pd.DataFrame(self.face_rows)
        cols = ["frame"] + [f"{n}_{c}" for n in FACE_ROI_NAMES for c in "rgb"]
        df[[c for c in cols if c in df.columns]].to_csv(OUT_CSV, index=False)
        OUT_META.write_text(json.dumps({
            "fps": round(self.fps, 4),
            "frames_total": self.frame_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
        print(f"[Session] Face CSV exporté ({self.frame_count} frames, fps={self.fps:.2f})")


# ── FastAPI ───────────────────────────────────────────────────────────

app = FastAPI()

cnn1d_model:   Optional[object] = None
active_session: Optional[Session] = None


@app.on_event("startup")
def startup():
    global cnn1d_model
    cnn1d_model = load_cnn1d()


@app.get("/")
def index():
    return FileResponse(str(HTML_PATH)) if HTML_PATH.exists() else HTMLResponse("<h1>rppg_live.html introuvable</h1>")


@app.get("/download/{filename}")
def download(filename: str):
    p = HERE / filename
    return FileResponse(str(p), filename=filename) if p.exists() else HTMLResponse("Introuvable", status_code=404)


@app.get("/analyse")
def analyse_face_endpoint():
    if not OUT_CSV.exists():
        return JSONResponse({"error": "CSV introuvable — lancez une capture d'abord"}, status_code=404)
    try:
        sys.path.insert(0, str(ANALYSE_DIR))
        from AnalyseWebRPPG import analyse  # type: ignore
        df = pd.read_csv(OUT_CSV)
        meta = json.loads(OUT_META.read_text())
        df_results, bpm_final, votes, snr_moyen, coh_moyen = analyse(df, meta["fps"])
        return JSONResponse({
            "bpm_consensus": bpm_final,
            "bpm_votes":     votes,
            "bpm_snr":       snr_moyen,
            "bpm_coherence": coh_moyen,
            "roi_results":   df_results.to_dict(orient="records"),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


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
        ckpt   = str(CKPT_PATH)
        print(f"[process_video] subprocess inference_paume_video.py sur {tmp_path}…")
        proc = await asyncio.get_event_loop().run_in_executor(None, lambda: subprocess.run(
            [sys.executable, str(script),
             "--video", tmp_path,
             "--checkpoint", ckpt,
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
                        "status":      "done",
                        "frames":      session.frame_count,
                        "cnn1d_ready": session.cnn1d_result is not None,
                    }))
            elif "bytes" in msg:
                session.push_frame(msg["bytes"])
                await ws.send_text(json.dumps({
                    "status":  "processing",
                    "queued":  session.frame_queue.qsize(),
                    "done":    session.frame_count,
                    "palm_g":  session.last_palm_g,
                }))
    except WebSocketDisconnect:
        print("[WS] Client déconnecté")
    except Exception as e:
        print(f"[WS] Erreur : {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
