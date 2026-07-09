"""
Inférence CNN1D paume sur une vidéo INCONNUE (pas dans le dataset, sans GT).

Pipeline
--------
1. Extraction RGB multi-ROI (MediaPipe Hands + HSV) sur toute la vidéo
2. CNN1D → signal rPPG reconstruit
3. Calcul des healthmarks (HR, HRV, SpO2, RR) via metrics.py

Usage
-----
python inference_paume_video.py --video /chemin/video_paume.mp4 \
    --checkpoint checkpoints_paume_cnn1d/best.pth
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from cnn1d_model import CNN1D_rPPG
from extract_rgb_paume_multi import extract_rgb_paume_multi
from metrics import extract_all_metrics


def suppress_outlier_peaks(signal: np.ndarray, k: float = 3.5) -> np.ndarray:
    med = np.median(signal)
    mad = np.median(np.abs(signal - med))
    if mad < 1e-8:
        return signal
    return np.clip(signal, med - k * mad, med + k * mad)


def run_cnn1d_full(arr: np.ndarray, model, fps: float, device, clip_len: int = 512):
    """
    Inférence par clips chevauchants sur toute la vidéo, reconstruction
    du signal complet par fenêtre de Hann (comme inference_physnet.py).
    """
    T = arr.shape[0]

    # Interpolation NaN
    arr = arr.copy()
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
    arr_n = (arr - mu) / std

    stride = clip_len // 2
    rppg_full = np.zeros(T, dtype=np.float64)
    weight    = np.zeros(T, dtype=np.float64)
    window    = np.hanning(clip_len)

    starts = list(range(0, max(1, T - clip_len + 1), stride))
    if T >= clip_len and (T - clip_len) not in starts:
        starts.append(T - clip_len)

    model.eval()
    with torch.no_grad():
        for start in starts:
            end = min(start + clip_len, T)
            clip = arr_n[start:end]
            if len(clip) < clip_len:
                pad  = np.zeros((clip_len - len(clip), arr_n.shape[1]), dtype=np.float32)
                clip = np.concatenate([clip, pad])
            x = torch.from_numpy(clip.astype(np.float32).T).unsqueeze(0).to(device)
            rppg, _ = model(x)
            sig = rppg.squeeze(0).cpu().numpy()
            sig_std = sig.std()
            if sig_std > 1e-6:
                sig = (sig - sig.mean()) / sig_std
            L   = end - start
            rppg_full[start:end] += sig[:L] * window[:L]
            weight[start:end]    += window[:L]

    mask = weight > 0
    rppg_full[mask] /= weight[mask]
    return rppg_full


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")

    # Modèle
    model = CNN1D_rPPG(in_channels=12).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Checkpoint chargé (époque {ckpt['epoch']})")

    # Extraction RGB multi-ROI
    print("\nExtraction RGB multi-ROI (paume)...")
    df, fps = extract_rgb_paume_multi(args.video)
    arr = df.drop(columns=["frame"]).values.astype(np.float32)   # (T, 12)

    # Signal RGB brut moyen (pour SpO2) — moyenne des 4 ROI, canal R et B
    rgb_raw = arr.reshape(arr.shape[0], 4, 3).mean(axis=1)   # (T, 3) R,G,B moyen

    # Inférence CNN1D
    print("\nInférence CNN1D...")
    rppg = suppress_outlier_peaks(run_cnn1d_full(arr, model, fps, device, args.clip_len))

    # Healthmarks
    print("\nCalcul des healthmarks...")
    metrics = extract_all_metrics(rppg, fps, rgb_raw=rgb_raw)

    # Affichage
    print("\n" + "=" * 50)
    print("  RÉSULTATS — CNN1D PAUME")
    print("=" * 50)
    labels = {
        "hr_bpm":           "Fréquence cardiaque",
        "hrv_sdnn":         "HRV — SDNN",
        "hrv_rmssd":        "HRV — RMSSD",
        "hrv_lf_hf":        "HRV — LF/HF",
        "spo2_pct":         "SpO₂",
        "rr_bpm":           "Fréquence respiratoire",
        "ppg_rise_time_ms": "PPG — temps de montée",
        "ppg_aix":          "PPG — AIx",
        "n_beats":          "Battements détectés",
    }
    units = {
        "hr_bpm": "bpm", "hrv_sdnn": "ms", "hrv_rmssd": "ms",
        "hrv_lf_hf": "", "spo2_pct": "%", "rr_bpm": "resp/min",
        "ppg_rise_time_ms": "ms", "ppg_aix": "%", "n_beats": "",
    }
    for key, label in labels.items():
        print(f"  {label:35s} : {metrics.get(key, 'N/A')} {units.get(key,'')}")
    print("=" * 50)

    if args.out_json:
        import json as _json
        N = len(rppg)
        freqs = np.fft.rfftfreq(N, d=1.0/fps)
        spec  = np.abs(np.fft.rfft(rppg))
        mask  = (freqs >= 0.7) & (freqs <= 3.0)
        freqs_m, spec_m = freqs[mask], spec[mask]
        step = max(1, N // 600)
        payload = {
            "rppg_signal": rppg[::step].tolist(),
            "freqs_bpm":   (freqs_m * 60).tolist(),
            "spectrum":    (spec_m / (spec_m.max() + 1e-9)).tolist(),
            "hr_bpm":      round(float(metrics.get("hr_bpm", float("nan"))), 1),
            "n_frames":    N,
            "fps":         round(fps, 3),
        }
        Path(args.out_json).write_text(_json.dumps(payload))

    if args.plot:
        _plot(rppg, metrics, fps, args.video)


def _plot(rppg, metrics, fps, title):
    t       = np.arange(len(rppg)) / fps
    peaks_t = metrics.get("peaks_t", [])

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    axes[0].plot(t, rppg, linewidth=0.7, color="steelblue")
    for pt in peaks_t:
        axes[0].axvline(pt, color="tomato", linewidth=0.5, alpha=0.6)
    axes[0].set_title(
        f"Signal rPPG (paume) — HR : {metrics['hr_bpm']} bpm | SpO₂ : {metrics['spo2_pct']} %")
    axes[0].set_xlabel("Temps (s)")
    axes[0].set_ylabel("Amplitude (norm.)")
    axes[0].grid(True, alpha=0.3)

    N     = len(rppg)
    freqs = np.fft.rfftfreq(N, d=1.0/fps)
    spec  = np.abs(np.fft.rfft(rppg))
    mask  = (freqs >= 0.5) & (freqs <= 4.0)
    axes[1].plot(freqs[mask]*60, spec[mask]/spec[mask].max(), color="darkgreen", lw=0.9)
    if not np.isnan(metrics["hr_bpm"]):
        axes[1].axvline(metrics["hr_bpm"], color="tomato", linestyle="--", lw=1.5,
                        label=f"HR = {metrics['hr_bpm']} bpm")
    axes[1].set_xlabel("Fréquence (bpm)")
    axes[1].set_ylabel("Amplitude norm.")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"CNN1D Paume — {Path(title).name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints_paume_cnn1d/best.pth")
    parser.add_argument("--clip_len",   type=int, default=512)
    parser.add_argument("--plot",       action="store_true")
    parser.add_argument("--out_json",   type=str, default=None)
    args = parser.parse_args()
    main(args)
