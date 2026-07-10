"""
Extraction RGB multi-ROI sur la paume — extension de
semaine 1/ExtractionRGB_paume.py pour avoir plusieurs zones distinctes
(comme les 6 ROI du visage), nécessaire pour le CNN1D.

Détection : MediaPipe Hands (21 landmarks), équivalent de FaceMesh.
Segmentation peau : masque HSV, équivalent de BiSeNet (pas de modèle
de segmentation de peau spécifique aux mains disponible).

ROI définies (4 zones anatomiques de la paume)
------------------------------------------------
  centre       : paume entière (landmarks 0,5,9,13,17) — référence
  thenar       : éminence thénar, base du pouce (landmarks 0,1,2,5)
  hypothenar   : éminence hypothénar, côté auriculaire (landmarks 0,13,17)
  base_doigts  : jonction des phalanges (landmarks 5,9,13,17)

CSV produit : frame, centre_r, centre_g, centre_b, thenar_r, ..., base_doigts_b
              (4 ROI × 3 canaux = 12 colonnes + frame)

Usage
-----
python extract_rgb_paume_multi.py --video /chemin/video.mp4
"""

import argparse
import json
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# ----------------------------------------------------------------
# MediaPipe Hands
# ----------------------------------------------------------------

mp_hands = mp.solutions.hands

# Landmarks MediaPipe Hands :
#   0=poignet, 1-4=pouce, 5-8=index, 9-12=majeur, 13-16=annulaire, 17-20=auriculaire
PALM_ROIS = {
    "centre":      [0, 5, 9, 13, 17],
    "thenar":      [0, 1, 2, 5],
    "hypothenar":  [0, 13, 17],
    "base_doigts": [5, 9, 13, 17],
}
ROI_NAMES = list(PALM_ROIS.keys())


# ----------------------------------------------------------------
# Masques polygones par ROI
# ----------------------------------------------------------------

def roi_mask(shape, landmarks, indices):
    h, w = shape[:2]
    pts = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
        dtype=np.int32,
    )
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


# ----------------------------------------------------------------
# Masque peau HSV — remplace BiSeNet (pas de modèle pour les mains)
# ----------------------------------------------------------------

def skin_hsv_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0,  15, 40],  dtype=np.uint8)
    upper = np.array([25, 200, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower, upper)
    lower2 = np.array([165, 15, 40],  dtype=np.uint8)
    upper2 = np.array([180, 200, 255], dtype=np.uint8)
    mask2  = cv2.inRange(hsv, lower2, upper2)
    skin   = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)


# ----------------------------------------------------------------
# Extraction RGB par ROI
# ----------------------------------------------------------------

def region_rgb(frame_bgr, mask_roi, mask_skin, sigma=2.0):
    final_mask = cv2.bitwise_and(mask_roi, mask_skin)
    pixels = frame_bgr[final_mask > 0].astype(np.float32)

    if len(pixels) < 30:
        return np.nan, np.nan, np.nan

    keep = np.ones(len(pixels), dtype=bool)
    for c in range(3):
        med = np.median(pixels[:, c])
        std = np.std(pixels[:, c])
        if std > 0:
            keep &= np.abs(pixels[:, c] - med) <= sigma * std

    pixels = pixels[keep]
    if len(pixels) < 30:
        return np.nan, np.nan, np.nan

    b = np.mean(pixels[:, 0])
    g = np.mean(pixels[:, 1])
    r = np.mean(pixels[:, 2])
    return r, g, b


# ----------------------------------------------------------------
# Rotation (identique à ExtractionRGB_paume.py)
# ----------------------------------------------------------------

def _get_rotation_metadata(cap, video_path=None):
    if video_path:
        try:
            import subprocess
            out = subprocess.check_output(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
                stderr=subprocess.DEVNULL,
            )
            for s in json.loads(out).get("streams", []):
                tags = s.get("tags", {})
                rotate = tags.get("rotate") or tags.get("rotation")
                if rotate is not None:
                    return int(rotate)
                for sd in s.get("side_data_list", []):
                    if "rotation" in sd:
                        return int(sd["rotation"])
        except Exception:
            pass
    angle = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    return int(angle) if angle else 0


def _rotate_frame(frame, angle):
    angle = angle % 360
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


# ----------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------

def extract_rgb_paume_multi(video_path: str):
    t_start = time.perf_counter()

    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 29.97
        print("Avertissement : fps non détecté, valeur par défaut 29.97 utilisée")
    rotation = _get_rotation_metadata(cap, video_path)

    data = []
    frame_idx = 0
    hand_detected = 0
    usable_per_roi = {name: 0 for name in ROI_NAMES}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = _rotate_frame(frame, rotation)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)

        if not results.multi_hand_landmarks:
            data.append([frame_idx] + [np.nan] * (3 * len(ROI_NAMES)))
            frame_idx += 1
            continue

        hand_detected += 1
        lm = results.multi_hand_landmarks[0].landmark
        mask_skin = skin_hsv_mask(frame)

        row = [frame_idx]
        for name in ROI_NAMES:
            mask_roi = roi_mask(frame.shape, lm, PALM_ROIS[name])
            r, g, b = region_rgb(frame, mask_roi, mask_skin)
            if not np.isnan(r):
                usable_per_roi[name] += 1
            row.extend([r, g, b])

        data.append(row)
        frame_idx += 1

    cap.release()

    columns = ["frame"]
    for name in ROI_NAMES:
        columns.extend([f"{name}_r", f"{name}_g", f"{name}_b"])
    df = pd.DataFrame(data, columns=columns)

    elapsed = time.perf_counter() - t_start
    print("\n--- STATS PIPELINE PAUME MULTI-ROI ---")
    print(f"Temps d'exécution : {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps)")
    print(f"Frames totales    : {frame_idx}")
    print(f"Main détectée     : {hand_detected}  ({100*hand_detected/max(frame_idx,1):.1f}%)")
    for name in ROI_NAMES:
        n = usable_per_roi[name]
        print(f"  {name:<12} exploitable : {n}  ({100*n/max(frame_idx,1):.1f}%)")
    print(f"FPS vidéo         : {fps:.3f}")

    return df, fps


# ----------------------------------------------------------------
# Exécution
# ----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="../data/input.mp4")
    parser.add_argument("--csv",   type=str, default="rppg_rgb_paume_multi.csv")
    parser.add_argument("--meta",  type=str, default="rppg_rgb_paume_multi_meta.json")
    args = parser.parse_args()

    df, fps = extract_rgb_paume_multi(args.video)
    print(df.head(10))

    df.to_csv(args.csv, index=False)
    with open(args.meta, "w") as f:
        json.dump({
            "fps": fps,
            "video": os.path.basename(args.video),
            "rois": ROI_NAMES,
        }, f, indent=2)

    print(f"\nCSV  : {args.csv}")
    print(f"JSON : {args.meta}")
