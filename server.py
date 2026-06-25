#!/usr/bin/env python3
"""
Serveur rPPG — reçoit les frames vidéo en temps réel via WebSocket,
applique BiSeNet + extraction RGB pendant la capture,
exporte rppg_rgb.csv et rppg_rgb_meta.json à la fin.

Dépendances :
    pip install fastapi uvicorn websockets torch torchvision pillow numpy pandas opencv-python-headless

Lancement :
    python server.py
    # puis ouvrir http://localhost:8000 dans le navigateur
"""

import asyncio
import io
import json
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Chemins ───────────────────────────────────────────────────────
HERE        = Path(__file__).parent
MODEL_PATH  = HERE / "79999_iter.pth"   # poids BiSeNet — à placer à côté du script
HTML_PATH   = HERE / "rppg_live.html"
OUT_CSV     = HERE / "rppg_rgb.csv"
OUT_META    = HERE / "rppg_rgb_meta.json"

# ── ROI — indices MediaPipe identiques à ExtractionRGB.py ─────────
ROI_NAMES = [
    "front", "joue_gauche", "joue_droite",
    "nez", "sous_oeil_gauche", "sous_oeil_droit"
]
ROI_IDX = [
    [103,67,109,10,338,297,332,333,168,104],
    [187,214,211,57,216,203,101,118,117,123],
    [411,434,431,287,436,423,330,347,346,352],
    [1,45,134,174,197,399,363,275],
    [24,23,22,121,47,100,119,228],
    [252,253,254,448,348,329,277,350],
]

# Labels BiSeNet peau (face parsing — 19 classes)
# 1=skin, 2=left_brow, 3=right_brow, 4=left_eye, 5=right_eye,
# 6=glasses, 10=nose, 11=teeth, 12=upper_lip, 13=lower_lip
SKIN_LABELS = {1, 2, 3, 4, 5, 10, 12, 13}

# ── Chargement BiSeNet ────────────────────────────────────────────
def load_bisenet():
    """Charge le modèle BiSeNet depuis model.py/resnet.py locaux."""
    import sys
    sys.path.insert(0, str(HERE))
    from model import BiSeNet  # type: ignore

    net = BiSeNet(n_classes=19)
    if MODEL_PATH.exists():
        state = torch.load(str(MODEL_PATH), map_location="cpu")
        net.load_state_dict(state)
        print(f"[BiSeNet] Poids chargés depuis {MODEL_PATH}")
    else:
        print(f"[BiSeNet] ATTENTION : {MODEL_PATH} introuvable — masque désactivé")
    net.eval()
    return net

# ── Extraction RGB avec masque BiSeNet ───────────────────────────
def get_skin_mask(net, frame_bgr):
    """
    Retourne un masque binaire (H, W) où True = pixel de peau.
    frame_bgr : numpy array BGR uint8.
    """
    h, w = frame_bgr.shape[:2]
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = (img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        out, _, _ = net(tensor)
    pred = out.argmax(1).squeeze(0).numpy().astype(np.uint8)  # (512, 512)
    mask512 = np.isin(pred, list(SKIN_LABELS))
    mask = cv2.resize(mask512.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def point_in_poly(px, py, poly):
    """Ray-casting point-in-polygon."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def extract_rgb(frame_bgr, landmarks_norm, skin_mask):
    """
    Extrait la moyenne RGB par ROI avec masque peau.
    landmarks_norm : liste de {x, y} normalisés 0-1.
    Retourne dict {roi_name: {r, g, b}}.
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = {}

    for name, idx in zip(ROI_NAMES, ROI_IDX):
        pts = [(landmarks_norm[i]["x"] * w, landmarks_norm[i]["y"] * h) for i in idx]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = max(0, int(min(xs))), min(w, int(max(xs)) + 1)
        y0, y1 = max(0, int(min(ys))), min(h, int(max(ys)) + 1)

        rs, gs, bs, n = 0.0, 0.0, 0.0, 0
        for py in range(y0, y1):
            for px in range(x0, x1):
                if skin_mask[py, px] and point_in_poly(px, py, pts):
                    r, g, b = frame_rgb[py, px]
                    rs += r; gs += g; bs += b; n += 1

        if n > 0:
            result[name] = {"r": rs/n, "g": gs/n, "b": bs/n}
        else:
            result[name] = {"r": float("nan"), "g": float("nan"), "b": float("nan")}

    return result


# ── Gestionnaire de session WebSocket ────────────────────────────
class Session:
    def __init__(self):
        self.frame_queue: queue.Queue = queue.Queue()  # illimitée
        self.rows: list = []
        self.done = threading.Event()
        self.fps: float = 30.0
        self.bisenet = None
        self.worker = None
        self.frame_count: int = 0
        self.ws_send_cb = None  # callback async pour envoyer au client

    def start(self, bisenet, fps: float, ws_send_cb):
        self.bisenet = bisenet
        self.fps = fps
        self.ws_send_cb = ws_send_cb
        self.done.clear()
        self.rows = []
        self.frame_count = 0
        self.worker = threading.Thread(target=self._process_loop, daemon=True)
        self.worker.start()
        print(f"[Session] Démarrée — fps={fps}")

    def push_frame(self, jpeg_bytes: bytes, landmarks: list):
        self.frame_queue.put((jpeg_bytes, landmarks))

    def end(self):
        self.frame_queue.put(None)  # poison pill
        print("[Session] Signal END reçu — attente fin du worker…")

    def _process_loop(self):
        while True:
            item = self.frame_queue.get()
            if item is None:
                break
            jpeg_bytes, landmarks = item

            # Décodage JPEG
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            # Masque BiSeNet
            if self.bisenet is not None:
                try:
                    skin_mask = get_skin_mask(self.bisenet, frame_bgr)
                except Exception as e:
                    print(f"[BiSeNet] Erreur : {e}")
                    skin_mask = np.ones(frame_bgr.shape[:2], dtype=bool)
            else:
                skin_mask = np.ones(frame_bgr.shape[:2], dtype=bool)

            # Extraction RGB
            rgb = extract_rgb(frame_bgr, landmarks, skin_mask)
            row = {"frame": self.frame_count}
            for name in ROI_NAMES:
                v = rgb[name]
                row[f"{name}_r"] = v["r"]
                row[f"{name}_g"] = v["g"]
                row[f"{name}_b"] = v["b"]
            self.rows.append(row)
            self.frame_count += 1

            if self.frame_count % 30 == 0:
                print(f"[Session] {self.frame_count} frames traitées")

        # Fin — export CSV
        self._export()
        self.done.set()
        print("[Session] Terminée.")

    def _export(self):
        df = pd.DataFrame(self.rows)
        cols = ["frame"]
        for name in ROI_NAMES:
            cols += [f"{name}_r", f"{name}_g", f"{name}_b"]
        df = df[cols]
        df.to_csv(OUT_CSV, index=False)

        meta = {
            "fps": self.fps,
            "frames_total": self.frame_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        OUT_META.write_text(json.dumps(meta, indent=2))
        print(f"[Session] Exporté : {OUT_CSV} ({self.frame_count} frames)")


# ── App FastAPI ───────────────────────────────────────────────────
from typing import Optional

app = FastAPI()

# CORS — nécessaire pour que Lovable (autre domaine) puisse télécharger
# le CSV via /download/... . Le WebSocket lui-même n'a pas besoin de CORS,
# mais les requêtes GET classiques (fetch, <a href>) en ont besoin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # à restreindre à votre domaine Lovable en prod si besoin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
bisenet_model = None
active_session: Optional[Session] = None


@app.on_event("startup")
def startup():
    global bisenet_model
    bisenet_model = load_bisenet()


@app.get("/")
def index():
    if HTML_PATH.exists():
        return FileResponse(str(HTML_PATH))
    return HTMLResponse("<h1>rppg_live.html introuvable</h1>")


@app.get("/download/{filename}")
def download(filename: str):
    path = HERE / filename
    if path.exists():
        return FileResponse(str(path), filename=filename)
    return HTMLResponse("Fichier non trouvé", status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global active_session
    await ws.accept()
    print("[WS] Client connecté")

    session = Session()
    active_session = session

    loop = asyncio.get_event_loop()

    try:
        while True:
            msg = await ws.receive()

            # Message texte : JSON de contrôle
            if "text" in msg:
                data = json.loads(msg["text"])

                if data.get("type") == "START":
                    fps = float(data.get("fps", 30))
                    async def send_cb(obj):
                        try:
                            await ws.send_text(json.dumps(obj))
                        except Exception:
                            pass
                    session.start(bisenet_model, fps, send_cb)
                    await ws.send_text(json.dumps({"status": "started"}))

                elif data.get("type") == "FRAME_META":
                    # Landmarks associés à la prochaine frame binaire
                    session._pending_landmarks = data.get("landmarks", [])

                elif data.get("type") == "END":
                    session.end()
                    # Attendre la fin du processing dans un thread
                    await loop.run_in_executor(None, session.done.wait, 600)
                    await ws.send_text(json.dumps({
                        "status": "done",
                        "frames": session.frame_count,
                        "csv_url": "/download/rppg_rgb.csv",
                        "meta_url": "/download/rppg_rgb_meta.json",
                    }))

            # Message binaire : frame JPEG
            elif "bytes" in msg:
                landmarks = getattr(session, "_pending_landmarks", [])
                session.push_frame(msg["bytes"], landmarks)
                await ws.send_text(json.dumps({
                    "status": "processing",
                    "queued": session.frame_queue.qsize(),
                    "done": session.frame_count,
                }))

    except WebSocketDisconnect:
        print("[WS] Client déconnecté")
    except Exception as e:
        print(f"[WS] Erreur : {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
