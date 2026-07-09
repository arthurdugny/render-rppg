"""
Extraction des healthmarks à partir d'un signal rPPG reconstruit.

Fonctions publiques
-------------------
compute_snr_mask      : SNR spectral par fenêtre glissante → masque de qualité
extract_beats         : segmentation + normalisation par battement (filtrés par SNR)
morphology_from_beats : features morphologiques sur le template moyen
extract_all_metrics   : pipeline complet HR / HRV / SpO2 / RR / morphologie
"""

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d, CubicSpline


# ---------------------------------------------------------------------------
# Filtrage
# ---------------------------------------------------------------------------

def _bandpass(signal: np.ndarray, fps: float,
              low: float = 0.5, high: float = 4.0) -> np.ndarray:
    nyq = fps / 2.0
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# SNR spectral par fenêtre glissante
# ---------------------------------------------------------------------------

def compute_snr_mask(
    signal: np.ndarray,
    fps: float,
    win_s: float = 3.0,
    overlap: float = 0.5,
    threshold: float = 0.5,
    cardiac_low: float = 0.7,
    cardiac_high: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retourne (t_centers, snr_curve, mask).
    mask[i] = True si snr_curve[i] >= threshold.
    """
    T       = len(signal)
    win_len = int(win_s * fps)
    stride  = max(1, int(win_len * (1.0 - overlap)))

    t_centers, snr_vals = [], []

    for start in range(0, T - win_len + 1, stride):
        end  = start + win_len
        seg  = signal[start:end]
        freqs   = np.fft.rfftfreq(win_len, d=1.0 / fps)
        spec    = np.abs(np.fft.rfft(seg)) ** 2
        cardiac = (freqs >= cardiac_low) & (freqs <= cardiac_high)
        snr     = float(spec[cardiac].sum() / (spec.sum() + 1e-8))
        t_centers.append((start + end) / 2.0 / fps)
        snr_vals.append(snr)

    t_centers = np.array(t_centers)
    snr_curve = np.array(snr_vals)
    return t_centers, snr_curve, snr_curve >= threshold


# ---------------------------------------------------------------------------
# Fréquence cardiaque
# ---------------------------------------------------------------------------

def compute_spectrum(seg: np.ndarray, fps: float,
                     zero_pad: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (freqs_bpm, spec) du signal zero-paddé."""
    N     = len(seg) * zero_pad
    freqs = np.fft.rfftfreq(N, d=1.0 / fps) * 60.0
    spec  = np.abs(np.fft.rfft(seg, n=N))
    return freqs, spec


def _spectral_hr_from_spectrum(freqs: np.ndarray, spec: np.ndarray,
                                low: float = 0.7, high: float = 3.0) -> float:
    """Pic dominant dans la bande cardiaque sur un spectre déjà calculé."""
    mask = (freqs >= low * 60) & (freqs <= high * 60)
    if not mask.any():
        return np.nan
    f, s = freqs[mask], spec[mask]
    idx  = int(s.argmax())
    if 0 < idx < len(s) - 1:
        denom = s[idx-1] - 2*s[idx] + s[idx+1]
        if abs(denom) > 1e-10:
            delta = 0.5 * (s[idx-1] - s[idx+1]) / denom
            return float(f[idx] + delta * (f[1] - f[0]))
    return float(f[idx])


def _spectral_hr_segment(seg: np.ndarray, fps: float,
                          low: float = 0.7, high: float = 3.0) -> float:
    freqs, spec = compute_spectrum(seg, fps)
    return _spectral_hr_from_spectrum(freqs, spec, low, high)


def estimate_hr(signal: np.ndarray, fps: float,
                win_s: float = 5.0, overlap: float = 0.5,
                snr_mask: np.ndarray | None = None,
                snr_t:    np.ndarray | None = None,
                snr_global_threshold: float = 0.4) -> float:
    """
    Si le SNR moyen global est suffisant (>= snr_global_threshold) :
        HR sur le spectre FFT du signal entier (meilleure résolution).
    Sinon fallback :
        médiane des HR spectrales sur les fenêtres retenues (>= 6 requises).
    """
    good_ratio = float(snr_mask.mean()) if snr_mask is not None and len(snr_mask) > 0 else 0.0

    if good_ratio >= snr_global_threshold:
        freqs, spec = compute_spectrum(signal, fps)
        return _spectral_hr_from_spectrum(freqs, spec)

    # Fallback fenêtres retenues
    T, win_len = len(signal), int(win_s * fps)
    stride = max(1, int(win_len * (1.0 - overlap)))
    hrs = []
    for start in range(0, T - win_len + 1, stride):
        end = start + win_len
        t_c = (start + end) / 2.0 / fps
        if snr_mask is not None and snr_t is not None and len(snr_t) > 0:
            if not snr_mask[np.argmin(np.abs(snr_t - t_c))]:
                continue
        hr = _spectral_hr_segment(signal[start:end], fps)
        if not np.isnan(hr):
            hrs.append(hr)
    if len(hrs) < 6:
        return np.nan
    return float(np.median(hrs))


# ---------------------------------------------------------------------------
# Détection des pics systoliques
# ---------------------------------------------------------------------------

def detect_peaks(signal: np.ndarray, fps: float) -> np.ndarray:
    sig_f    = _bandpass(signal, fps, 0.5, 4.0)
    min_dist = max(1, int(fps * 0.33))   # 180 bpm max
    peaks, _ = find_peaks(sig_f, distance=min_dist,
                           prominence=0.1 * (sig_f.max() - sig_f.min()))
    return peaks


def _refine_peaks_xcorr(signal: np.ndarray, peaks: np.ndarray,
                         fps: float, search_ms: float = 40.0) -> np.ndarray:
    """
    Raffine les positions de pics par cross-corrélation avec le template moyen.
    Pour chaque pic, cherche dans ±search_ms ms la position qui maximise
    la corrélation avec le template PPG moyen (résolution native).
    Réduit le jitter de détection et améliore la précision des intervalles RR.
    """
    if len(peaks) < 3:
        return peaks

    search = max(1, int(search_ms / 1000.0 * fps))
    rr_med = int(np.median(np.diff(peaks)))
    half   = max(search + 1, rr_med // 2)

    # Template moyen centré sur chaque pic (résolution native)
    segs = []
    for p in peaks:
        if p - half < 0 or p + half >= len(signal):
            continue
        seg = signal[p - half : p + half]
        segs.append((seg - seg.mean()) / (seg.std() + 1e-6))
    if len(segs) < 2:
        return peaks

    tpl = np.mean(segs, axis=0)

    # Raffiner chaque pic
    refined = []
    for p in peaks:
        s = p - half - search
        e = p + half + search
        if s < 0 or e > len(signal):
            refined.append(p)
            continue
        window      = signal[s:e]
        window_norm = (window - window.mean()) / (window.std() + 1e-6)
        xcorr       = np.correlate(window_norm, tpl, mode="valid")
        if len(xcorr) == 0:
            refined.append(p)
            continue
        lag   = int(np.argmax(xcorr))
        p_new = int(np.clip(s + lag + half, 0, len(signal) - 1))
        refined.append(p_new)

    return np.array(refined, dtype=int)


def rr_intervals_ms(peaks: np.ndarray, fps: float) -> np.ndarray:
    return np.diff(peaks) / fps * 1000.0 if len(peaks) >= 2 else np.array([])


# ---------------------------------------------------------------------------
# Étape 2+3 — Segmentation par battement + normalisation
# ---------------------------------------------------------------------------

def _find_onset(signal: np.ndarray, peak_idx: int, prev_peak_idx: int) -> int:
    """
    Cherche le pied (onset) du battement entre prev_peak et peak :
    minimum local du signal filtré dans cette fenêtre.
    """
    seg = signal[prev_peak_idx:peak_idx]
    return int(prev_peak_idx + np.argmin(seg))


def _normalize_beat(beat: np.ndarray, n_points: int = 100) -> np.ndarray | None:
    if len(beat) < 4:
        return None
    x_old = np.linspace(0, 1, len(beat))
    x_new = np.linspace(0, 1, n_points)
    b     = interp1d(x_old, beat, kind="linear")(x_new)
    lo, hi = b.min(), b.max()
    if hi - lo < 1e-6:
        return None
    return (b - lo) / (hi - lo)


def extract_beats(
    signal: np.ndarray,
    fps: float,
    snr_mask: np.ndarray | None = None,
    snr_t:    np.ndarray | None = None,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segmente le signal en battements individuels normalisés.

    Chaque battement = [onset[i], onset[i+1]]
    où onset est le minimum local entre deux pics systoliques consécutifs.

    Retourne
    --------
    beats     : (N, n_points) battements normalisés retenus
    peak_idxs : (N,) indices des pics systoliques correspondants
    """
    sig_f = _bandpass(signal, fps, 0.5, 4.0)
    peaks = detect_peaks(signal, fps)

    if len(peaks) < 3:
        return np.empty((0, n_points)), np.array([], dtype=int)

    # Onsets : minimum entre deux pics consécutifs
    onsets = [_find_onset(sig_f, peaks[i], peaks[i - 1])
              for i in range(1, len(peaks))]
    onsets = [_find_onset(sig_f, peaks[0], max(0, peaks[0] - int(fps)))] + onsets

    beats_norm, good_peaks = [], []

    for i in range(1, len(peaks)):
        t_c = peaks[i] / fps

        # Filtre SNR
        if snr_mask is not None and snr_t is not None and len(snr_t) > 0:
            if not snr_mask[np.argmin(np.abs(snr_t - t_c))]:
                continue

        onset_i = onsets[i - 1]
        onset_next = onsets[i] if i < len(onsets) else peaks[i] + int(fps * 0.3)
        beat = signal[onset_i:onset_next]

        b_norm = _normalize_beat(beat, n_points)
        if b_norm is None:
            continue

        beats_norm.append(b_norm)
        good_peaks.append(peaks[i])

    if not beats_norm:
        return np.empty((0, n_points)), np.array([], dtype=int)

    beats_arr = np.array(beats_norm)
    good_peaks = np.array(good_peaks, dtype=int)

    # Rejet MAD : écart au template médian > 2 MAD
    template = np.median(beats_arr, axis=0)
    dists    = np.mean(np.abs(beats_arr - template), axis=1)
    mad      = np.median(np.abs(dists - np.median(dists))) + 1e-8
    kept     = dists <= np.median(dists) + 2 * mad
    if kept.sum() < 2:
        kept = np.ones(len(beats_arr), dtype=bool)

    return beats_arr[kept], good_peaks[kept]


# ---------------------------------------------------------------------------
# Étape 4 — Features morphologiques sur le template moyen
# ---------------------------------------------------------------------------

def morphology_from_beats(
    beats: np.ndarray,
    fps: float,
    peak_idxs: np.ndarray | None = None,
    n_points: int = 100,
    min_beats: int = 5,
) -> dict:
    """
    Calcule les features morphologiques sur le template PPG moyen.

    Features
    --------
    ppg_rise_time_ms  : t(pic systolique) − t(onset) en ms
    ppg_width50_pct   : largeur à mi-hauteur (% du cycle RR)
    ppg_aix           : Augmentation Index = (P2−P1)/P1 × 100%
    ppg_sd_ratio      : surface systole / surface totale
    ppg_descent_slope : pente linéaire pic systolique → pied dicrote (unité/% cycle)
    n_beats           : nombre de battements retenus
    """
    nan_result = {
        "ppg_rise_time_ms":  np.nan,
        "ppg_width50_pct":   np.nan,
        "ppg_aix":           np.nan,
        "ppg_sd_ratio":      np.nan,
        "ppg_descent_slope": np.nan,
        "n_beats":           len(beats) if beats is not None else 0,
    }

    if beats is None or len(beats) < min_beats:
        return nan_result

    template = beats.mean(axis=0)          # (n_points,)
    x        = np.linspace(0, 100, n_points)

    # Upsampling C∞ par spline cubique (100 → 1000 points)
    n_up   = n_points * 10
    x_up   = np.linspace(0, 100, n_up)
    cs     = CubicSpline(x, template)
    tpl    = cs(x_up)          # template lissé haute résolution
    dtpl   = cs(x_up, 1)      # dérivée première
    d2tpl  = cs(x_up, 2)      # dérivée seconde

    # Pic systolique
    sys_idx = int(np.argmax(tpl))

    # --- Temps de montée ---
    if peak_idxs is not None and len(peak_idxs) >= 2:
        rr_mean_ms = float(np.mean(np.diff(peak_idxs)) / fps * 1000.0)
    else:
        rr_mean_ms = np.nan
    rise_time_ms = (sys_idx / n_up * rr_mean_ms) if not np.isnan(rr_mean_ms) else np.nan

    # --- Largeur à mi-hauteur ---
    above_half  = tpl >= 0.5
    width50_pct = float(above_half.sum() / n_up * 100.0)

    # --- Ratio systole/diastole (surface) ---
    area_total = np.trapezoid(tpl, x_up)
    area_sys   = np.trapezoid(tpl[:sys_idx + 1], x_up[:sys_idx + 1])
    sd_ratio   = float(area_sys / (area_total + 1e-8))

    # --- AIx et pente de descente (dérivée seconde spline, C∞) ---
    descent    = tpl[sys_idx:]
    x_desc     = x_up[sys_idx:]
    d2_desc    = d2tpl[sys_idx:]
    aix        = np.nan
    desc_slope = np.nan

    if len(descent) >= 4:
        # Inflexion positive dans la descente = pic dicrote
        candidates = np.where(
            (d2_desc[1:-1] > 0) & (descent[1:-1] < tpl[sys_idx])
        )[0]
        if len(candidates) > 0:
            dic_local = int(candidates[0]) + 1
            p1 = float(tpl[sys_idx])
            p2 = float(descent[dic_local])
            aix = round((p2 - p1) / (p1 + 1e-6) * 100.0, 1)

            dx = float(x_desc[dic_local] - x_desc[0])
            if dx > 0:
                desc_slope = round((p2 - p1) / dx, 4)

    return {
        "ppg_rise_time_ms":  round(rise_time_ms, 1) if not np.isnan(rise_time_ms) else np.nan,
        "ppg_width50_pct":   round(width50_pct, 1),
        "ppg_aix":           aix,
        "ppg_sd_ratio":      round(sd_ratio, 3),
        "ppg_descent_slope": desc_slope,
        "n_beats":           len(beats),
        "_template_hires":   tpl,    # spline C∞, 1000 points sur [0, 100%]
        "_x_hires":          x_up,
    }


# ---------------------------------------------------------------------------
# HRV
# ---------------------------------------------------------------------------

def compute_hrv(rr_ms: np.ndarray) -> dict:
    if len(rr_ms) < 2:
        return {"hrv_sdnn": np.nan, "hrv_rmssd": np.nan, "hrv_lf_hf": np.nan}

    sdnn  = float(np.std(rr_ms, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))

    lf_hf = np.nan
    if len(rr_ms) >= 8:
        t_rr   = np.cumsum(rr_ms) / 1000.0
        t_uni  = np.linspace(t_rr[0], t_rr[-1], len(rr_ms) * 4)
        rr_i   = interp1d(t_rr, rr_ms, kind="linear")(t_uni)
        fps_rr = len(t_uni) / (t_rr[-1] - t_rr[0])
        freqs  = np.fft.rfftfreq(len(rr_i), d=1.0 / fps_rr)
        psd    = np.abs(np.fft.rfft(rr_i - rr_i.mean())) ** 2
        lf     = psd[(freqs >= 0.04) & (freqs < 0.15)].sum()
        hf     = psd[(freqs >= 0.15) & (freqs < 0.40)].sum()
        lf_hf  = float(lf / (hf + 1e-8))

    return {
        "hrv_sdnn":  round(sdnn, 1),
        "hrv_rmssd": round(rmssd, 1),
        "hrv_lf_hf": round(lf_hf, 3) if not np.isnan(lf_hf) else np.nan,
    }


# ---------------------------------------------------------------------------
# SpO2
# ---------------------------------------------------------------------------

def estimate_spo2(rgb_raw: np.ndarray, fps: float,
                  snr_mask: np.ndarray | None = None,
                  snr_t:    np.ndarray | None = None) -> float:
    if rgb_raw is None or rgb_raw.shape[1] < 3:
        return np.nan
    R, B    = rgb_raw[:, 0].astype(np.float64), rgb_raw[:, 2].astype(np.float64)
    win_len = int(5.0 * fps)
    stride  = int(win_len * 0.5)
    ratios  = []
    for start in range(0, len(R) - win_len + 1, stride):
        end = start + win_len
        t_c = (start + end) / 2.0 / fps
        if snr_mask is not None and snr_t is not None and len(snr_t) > 0:
            if not snr_mask[np.argmin(np.abs(snr_t - t_c))]:
                continue
        r_seg, b_seg = R[start:end], B[start:end]
        dc_r, dc_b   = r_seg.mean(), b_seg.mean()
        if dc_r < 1e-3 or dc_b < 1e-3:
            continue
        ratios.append((r_seg.std() / dc_r) / (b_seg.std() / dc_b + 1e-8))
    if not ratios:
        return np.nan
    spo2 = 110.0 - 25.0 * float(np.median(ratios))
    return round(float(np.clip(spo2, 85.0, 100.0)), 1)


# ---------------------------------------------------------------------------
# Fréquence respiratoire
# ---------------------------------------------------------------------------

def estimate_rr(signal: np.ndarray, fps: float,
                snr_mask: np.ndarray | None = None,
                snr_t:    np.ndarray | None = None) -> float:
    peaks = detect_peaks(signal, fps)
    if len(peaks) < 6:
        return np.nan
    envelope = np.abs(signal[peaks])
    t_peaks  = peaks / fps
    fps_env  = 4.0
    t_uni    = np.arange(t_peaks[0], t_peaks[-1], 1.0 / fps_env)
    if len(t_uni) < 8:
        return np.nan
    env_i = interp1d(t_peaks, envelope, kind="linear",
                     bounds_error=False, fill_value="extrapolate")(t_uni)
    freqs = np.fft.rfftfreq(len(env_i), d=1.0 / fps_env) * 60.0
    spec  = np.abs(np.fft.rfft(env_i - env_i.mean()))
    mask  = (freqs >= 8) & (freqs <= 30)
    return round(float(freqs[mask][spec[mask].argmax()]), 1) if mask.any() else np.nan


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def _detrend_bandpass(signal: np.ndarray, fps: float) -> np.ndarray:
    """Filtre passe-bande [0.7–4 Hz] = [42–240 bpm], bande cardiaque stricte."""
    return _bandpass(signal, fps, low=0.7, high=4.0)


def extract_all_metrics(
    signal: np.ndarray,
    fps: float,
    rgb_raw: np.ndarray | None = None,
    snr_win_s:     float = 3.0,
    snr_overlap:   float = 0.5,
    snr_threshold: float = 0.5,
    n_points:      int   = 100,
    min_beats:     int   = 5,
) -> dict:
    """
    Pipeline complet :
    1. Masque SNR par fenêtres glissantes
    2. HR médiane sur bonnes fenêtres
    3. extract_beats → battements filtrés + normalisés
    4. HRV sur RR des battements retenus
    5. morphology_from_beats → features morphologiques
    6. SpO2 / RR respiratoire
    """
    signal_detrended = _detrend_bandpass(signal, fps)
    signal = signal_detrended
    freqs_global, spec_global = compute_spectrum(signal, fps)

    snr_t, snr_curve, snr_mask = compute_snr_mask(
        signal, fps, win_s=snr_win_s, overlap=snr_overlap, threshold=snr_threshold
    )
    good_ratio = float(snr_mask.mean()) if len(snr_mask) > 0 else 0.0

    hr = estimate_hr(signal, fps, win_s=5.0, overlap=0.5,
                     snr_mask=snr_mask, snr_t=snr_t)

    beats, peak_idxs = extract_beats(signal, fps,
                                     snr_mask=snr_mask, snr_t=snr_t,
                                     n_points=n_points)

    rr_ms = rr_intervals_ms(peak_idxs, fps)
    hrv   = compute_hrv(rr_ms)

    morph = morphology_from_beats(beats, fps, peak_idxs=peak_idxs,
                                  n_points=n_points, min_beats=min_beats)

    spo2 = estimate_spo2(rgb_raw, fps, snr_mask=snr_mask, snr_t=snr_t)
    rr   = estimate_rr(signal, fps, snr_mask=snr_mask, snr_t=snr_t)

    return {
        # Cardiovasculaire
        "hr_bpm":            round(hr, 1) if not np.isnan(hr) else np.nan,
        "hrv_sdnn":          hrv["hrv_sdnn"],
        "hrv_rmssd":         hrv["hrv_rmssd"],
        "hrv_lf_hf":         hrv["hrv_lf_hf"],
        "spo2_pct":          spo2,
        "rr_bpm":            rr,
        # Morphologie PPG
        "ppg_rise_time_ms":  morph["ppg_rise_time_ms"],
        "ppg_width50_pct":   morph["ppg_width50_pct"],
        "ppg_aix":           morph["ppg_aix"],
        "ppg_sd_ratio":      morph["ppg_sd_ratio"],
        "ppg_descent_slope": morph["ppg_descent_slope"],
        "n_beats":           morph["n_beats"],
        "peaks_t":           [p / fps for p in peak_idxs],
        # Signal détrendé + spectre (source unique pour affichage et HR)
        "signal_detrended":  signal_detrended,
        "freqs_global":      freqs_global,
        "spec_global":       spec_global,
        # Méta-qualité
        "snr_curve":         snr_curve,
        "snr_t":             snr_t,
        "snr_good_ratio":    round(good_ratio, 3),
        # Template PPG haute résolution (debug / visualisation)
        "_template":         beats.mean(axis=0) if len(beats) >= min_beats else None,
        "_template_hires":   morph.get("_template_hires"),
        "_x_hires":          morph.get("_x_hires"),
        "_beats":            beats,
    }
