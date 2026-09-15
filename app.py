"""
Accident Severity Classification API — v2 taxonomy, Phase 5 operating point.

Phase 5 (2026-09): model retrained on the features this service computes
(removing train/serve skew), crash and Severe thresholds selected on group-aware
out-of-fold predictions, Severe graded by P(Severe | crash) instead of argmax,
unit guard made robust to long high-g windows, NaN/Inf rejected with 400.
Evidence: phase5/PHASE5_REPORT.md.

Changes vs the Phase 1 production app:
  1. New taxonomy: severity is a CRASH SIGNATURE (2-7 g resultant, ~40-250 ms
     transient) graded by impulse (delta-v proxy) — NOT peak height.
  2. REMOVED the peak >= 7 g -> force-Severe escalation.  Phase 2 (VZCrash)
     proved >7 g events are maneuvers/artifacts, not crashes, and that real
     crashes are 2-7 g; the old rule escalated artifacts and missed real crashes.
  3. Model retrained on UNIT-NORMALIZED (g) data + VZCrash real crashes, and
     probability-CALIBRATED; confidence thresholds re-derived from the crash
     precision/recall trade-off (Task 6).
  4. Defensive per-request unit normalization to g (10th-pct magnitude > 5 -> m/s²)
     so a mis-configured client cannot re-introduce the unit-contamination bug.

Request contract UNCHANGED: POST /predict with ax,ay,az,gx,gy,gz arrays of 500
samples.  Response adds `p_crash` and `label_source`; `severity_class` semantics
now mean Normal(no crash) / Moderate(minor crash) / Severe(serious crash).
"""
import json, logging, os, time
from pathlib import Path
import joblib, numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from scipy import stats, signal

ART = Path(__file__).parent / "artifacts"
TARGET_FS, WINDOW_SAMPLES = 100, 500
CLASS_NAMES = ["Normal", "Moderate", "Severe"]

# ── crash-signature physics thresholds (from Phase 2 / VZCrash) ──────────────
PEAK_MIN_G, PEAK_MAX_G = 2.0, 7.0
# Phase 3 (Task 2) showed the 40 ms floor caps system recall at ~81%; lowering it
# to 20 ms raises recall to ~95% at a small false-positive cost. Left at 40 ms by
# default (conservative); set SIG_TRANSIENT_MIN_MS=20 to adopt the wider bound.
TRANSIENT_MIN_MS = float(os.environ.get("SIG_TRANSIENT_MIN_MS", "40"))
TRANSIENT_MAX_MS = 250.0
G = 9.80665
# Butterworth 20 Hz low-pass to match the harmonised training pipeline (Phase 3
# Task 1b). The ESP32 sends unfiltered g; filtering here aligns train/serve.
BUTTER_CUTOFF_HZ = 20.0
_SOS = signal.butter(4, BUTTER_CUTOFF_HZ / (TARGET_FS / 2.0), btype="low", output="sos")


def _lp(x):
    """Zero-phase 20 Hz low-pass; no-op if too short."""
    return signal.sosfiltfilt(_SOS, x) if len(x) > 15 else x

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Model + operating point come from ONE file, produced by phase5/p5_tune.py.
# No silent fallback: a missing or partial config must stop the service, because
# a default threshold quietly changes every classification.
_cfg = json.load(open(ART / "decision_config.json"))
MODEL_NAME = _cfg["model"]
_model = joblib.load(ART / _cfg["model_file"])
FEATURES = json.load(open(ART / _cfg["feature_names_file"]))
CRASH_ALERT_THRESHOLD = float(_cfg["crash_threshold"])
SEVERE_RATIO_THRESHOLD = float(_cfg["severe_ratio_threshold"])
_CAL_X = np.asarray(_cfg["pcrash_calibration"]["x"], float)
_CAL_Y = np.asarray(_cfg["pcrash_calibration"]["y"], float)


def calibrate(proba):
    """Isotonic P(crash) calibration; P(Moderate), P(Severe) rescaled so P(Severe | crash) is unchanged."""
    pc = proba[1] + proba[2]
    pcc = float(np.interp(pc, _CAL_X, _CAL_Y))
    s = pcc / pc if pc > 0 else 0.0
    return np.array([1.0 - pcc, proba[1] * s, proba[2] * s])
log.info("Model %s loaded (%d features); crash threshold P(crash)>=%.3f; "
         "Severe threshold P(Severe|crash)>=%.3f", MODEL_NAME, len(FEATURES),
         CRASH_ALERT_THRESHOLD, SEVERE_RATIO_THRESHOLD)

app = Flask(__name__)
CORS(app)


def _safe(f, a, d=0.0):
    try:
        v = float(f(a)); return v if np.isfinite(v) else d
    except Exception:
        return d


def _spec(a):
    v = np.abs(np.fft.rfft(a)); v[0] = 0.0
    return float(np.sum(v ** 2))


def normalize_to_g(ax, ay, az):
    """Defensive: convert accel to g if the window looks like m/s² (gravity ~9.8).

    Uses the 10th percentile of |a|, not the median: an m/s² window sits near
    9.8 almost throughout, whereas a g window only exceeds 5 during events. The
    median misfired on g windows with >2.5 s above 5 g (e.g. a sustained 9 g
    manoeuvre) and rescaled them by 1/9.81.
    """
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    scale = G if np.percentile(mag, 10) > 5.0 else 1.0
    return ax / scale, ay / scale, az / scale, scale


def features_25(ax, ay, az, gx):
    a_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    return {
        "ax_mean": ax.mean(), "ax_std": ax.std(), "ax_min": ax.min(), "ax_max": ax.max(),
        "ax_peak": np.abs(ax).max(), "ax_spectral_energy": _spec(ax),
        "ay_mean": ay.mean(), "ay_std": ay.std(), "ay_min": ay.min(), "ay_max": ay.max(),
        "ay_rms": np.sqrt((ay ** 2).mean()), "ay_kurt": _safe(stats.kurtosis, ay),
        "ay_peak": np.abs(ay).max(), "ay_spectral_energy": _spec(ay),
        "az_mean": az.mean(), "az_std": az.std(), "az_rms": np.sqrt((az ** 2).mean()),
        "az_spectral_energy": _spec(az),
        "a_mag_std": a_mag.std(), "a_mag_skew": _safe(stats.skew, a_mag),
        "a_mag_kurt": _safe(stats.kurtosis, a_mag), "a_mag_spectral_energy": _spec(a_mag),
        "gx_mean": gx.mean(), "gx_kurt": _safe(stats.kurtosis, gx),
        "mean_jerk": np.abs(np.diff(ax)).mean(),
    }


def crash_signature(ax, ay, az):
    """Physics cross-check: peak_g, longest excursion >2g (ms), impulse (g·s)."""
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    peak = float(mag.max())
    above = mag >= PEAK_MIN_G
    longest = run = 0
    for a in above:
        run = run + 1 if a else 0
        longest = max(longest, run)
    longest_ms = longest * 1000.0 / TARGET_FS
    impulse = float(np.sum(mag[above] - 1.0) / TARGET_FS) if above.any() else 0.0
    is_sig = (PEAK_MIN_G <= peak < PEAK_MAX_G and TRANSIENT_MIN_MS <= longest_ms <= TRANSIENT_MAX_MS)
    return peak, longest_ms, impulse, bool(is_sig)


def run_inference(ax, ay, az, gx, gy, gz):
    ax, ay, az, scale = normalize_to_g(np.asarray(ax, float), np.asarray(ay, float), np.asarray(az, float))
    ax, ay, az = _lp(ax), _lp(ay), _lp(az)          # 20 Hz LP to match training
    gx = np.asarray(gx, float)
    feat = features_25(ax, ay, az, gx)
    X = np.nan_to_num(np.array([[feat[n] for n in FEATURES]], float))
    proba = calibrate(_model.predict_proba(X)[0])
    model_label = int(np.argmax(proba))
    p_crash = float(proba[1] + proba[2])
    peak, longest_ms, impulse, is_sig = crash_signature(ax, ay, az)

    # ── decision: model P(crash) AND physics signature must agree ────────────
    # The model alone can call a brief >=7 g spike "Moderate"; the signature gate
    # (crash = 2-7 g, 40-250 ms transient) overrides such non-crash inputs to
    # Normal.  This is the corrected replacement for the old peak>=7g escalation
    # and is what fixes the brief-spike failure end-to-end.
    #
    # Grade: Severe when P(Severe | crash) clears a tuned threshold. The previous
    # argmax grade under-called Severe (the class is 7x rarer than Moderate):
    # 110 of 1,216 held-out Severe windows came back Moderate.
    model_says_crash = p_crash >= CRASH_ALERT_THRESHOLD
    if model_says_crash and is_sig:
        p_severe_given_crash = float(proba[2] / p_crash) if p_crash > 0 else 0.0
        label = 2 if p_severe_given_crash >= SEVERE_RATIO_THRESHOLD else 1
        label_source = "model+signature"
    else:
        label = 0                                   # physics override -> Normal
        label_source = ("signature_override" if model_says_crash
                        else "model")
    accident_confirmed = bool(label >= 1)
    return {
        "severity_class": label, "severity_name": CLASS_NAMES[label],
        "confidence": round(float(proba[label]), 4),
        "p_crash": round(p_crash, 4),
        "model_severity": CLASS_NAMES[model_label],
        "accident_confirmed": accident_confirmed,
        "probabilities": {CLASS_NAMES[i]: round(float(proba[i]), 4) for i in range(3)},
        "crash_signature": {"peak_g": round(peak, 3), "excursion_ms": round(longest_ms, 1),
                             "impulse_gs": round(impulse, 3), "signature_match": is_sig},
        "unit_scale_applied": scale, "label_source": label_source,
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": MODEL_NAME,
                    "n_features": len(FEATURES), "taxonomy": "signature+impulse (v2)",
                    "crash_alert_threshold": CRASH_ALERT_THRESHOLD,
                    "severe_ratio_threshold": SEVERE_RATIO_THRESHOLD,
                    "classes": CLASS_NAMES}), 200


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.perf_counter()
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    arrays = {}
    for f in ("ax", "ay", "az", "gx", "gy", "gz"):
        if f not in data:
            return jsonify({"error": f"Missing field: {f}"}), 400
        arr = np.array(data[f], float)
        if arr.shape != (WINDOW_SAMPLES,):
            return jsonify({"error": f"{f} must be {WINDOW_SAMPLES} samples"}), 400
        # NaN/Inf (e.g. a null from the device) would otherwise flow through
        # nan_to_num into all-zero features and a silent "Normal".
        if not np.all(np.isfinite(arr)):
            return jsonify({"error": f"Field '{f}' contains NaN or Inf values"}), 400
        arrays[f] = arr
    try:
        res = run_inference(**arrays)
    except Exception as exc:
        log.exception("inference error")
        return jsonify({"error": f"Inference failed: {exc}"}), 500
    res["inference_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return jsonify(res), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
