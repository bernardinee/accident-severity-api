"""
Accident Severity Classification API
Flask REST API wrapping the xgboost_no_proxy model (25 IMU features).

Input:  POST /predict  — JSON with ax, ay, az, gx, gy, gz arrays (500 samples @ 100 Hz)
Output: severity_class (0/1/2), confidence, accident_confirmed (dual-validation)
"""

import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR      = Path(__file__).parent / "models"
TARGET_FS      = 100
WINDOW_SAMPLES = 500
CLASS_NAMES    = ["Normal", "Moderate", "Severe"]
ACCIDENT_G_THRESHOLD  = 2.0   # physics dual-validation threshold (g)
SEVERE_G_THRESHOLD    = 7.0   # physics fallback boundary between Moderate/Severe (g)
CONFIDENCE_THRESHOLD  = 0.70  # below this, prefer physics thresholds over the model's label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Model loading (done once at startup) ──────────────────────────────────────
log.info("Loading model from %s ...", MODEL_DIR)
_model  = joblib.load(MODEL_DIR / "xgboost_no_proxy.joblib")
_scaler = joblib.load(MODEL_DIR / "scaler_no_proxy.joblib")
with open(MODEL_DIR / "feature_names_no_proxy.json") as f:
    FEATURE_NAMES = json.load(f)   # ordered list of 25 feature names
log.info("Model loaded — %d features, classes: %s", len(FEATURE_NAMES), CLASS_NAMES)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow cross-origin requests (ESP32, browser clients, etc.)


# ── Feature extraction ─────────────────────────────────────────────────────────
def _safe_stat(func, arr, default=0.0):
    """Compute a scalar stat; return default on error or non-finite result."""
    try:
        v = float(func(arr))
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _spectral_energy(arr):
    """Sum of squared FFT magnitudes with DC component zeroed."""
    fft_vals    = np.abs(np.fft.rfft(arr))
    fft_vals[0] = 0.0   # remove DC
    return float(np.sum(fft_vals ** 2))


def extract_features(ax, ay, az, gx):
    """
    Compute the 25 features used by xgboost_no_proxy from raw IMU arrays.

    Feature definitions replicate preprocess.py stage8_extract_features() exactly:
      - {col}_peak         = max(abs(col))
      - {col}_rms          = sqrt(mean(col^2))
      - {col}_kurt         = scipy excess kurtosis
      - {col}_spectral_energy = sum(|rfft(col)|^2)  [DC zeroed]
      - a_mag_*            = computed on sqrt(ax^2+ay^2+az^2)
      - mean_jerk          = mean(abs(diff(ax)))   [first available axis]

    Returns:
        feat      (dict)  : feature_name -> float
        peak_mag  (float) : peak resultant acceleration in g (for dual-validation)
    """
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    az = np.asarray(az, dtype=float)
    gx = np.asarray(gx, dtype=float)

    a_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    feat = {
        # ── ax ──────────────────────────────────────────────────────────────
        "ax_mean":              float(np.mean(ax)),
        "ax_std":               float(np.std(ax)),
        "ax_min":               float(np.min(ax)),
        "ax_max":               float(np.max(ax)),
        "ax_peak":              float(np.max(np.abs(ax))),
        "ax_spectral_energy":   _spectral_energy(ax),
        # ── ay ──────────────────────────────────────────────────────────────
        "ay_mean":              float(np.mean(ay)),
        "ay_std":               float(np.std(ay)),
        "ay_min":               float(np.min(ay)),
        "ay_max":               float(np.max(ay)),
        "ay_rms":               float(np.sqrt(np.mean(ay ** 2))),
        "ay_kurt":              _safe_stat(stats.kurtosis, ay),
        "ay_peak":              float(np.max(np.abs(ay))),
        "ay_spectral_energy":   _spectral_energy(ay),
        # ── az ──────────────────────────────────────────────────────────────
        "az_mean":              float(np.mean(az)),
        "az_std":               float(np.std(az)),
        "az_rms":               float(np.sqrt(np.mean(az ** 2))),
        "az_spectral_energy":   _spectral_energy(az),
        # ── resultant magnitude ──────────────────────────────────────────────
        "a_mag_std":            float(np.std(a_mag)),
        "a_mag_skew":           _safe_stat(stats.skew, a_mag),
        "a_mag_kurt":           _safe_stat(stats.kurtosis, a_mag),
        "a_mag_spectral_energy": _spectral_energy(a_mag),
        # ── gyroscope ────────────────────────────────────────────────────────
        "gx_mean":              float(np.mean(gx)),
        "gx_kurt":              _safe_stat(stats.kurtosis, gx),
        # ── jerk (mean abs first difference of ax) ───────────────────────────
        "mean_jerk":            float(np.mean(np.abs(np.diff(ax)))),
    }

    return feat, float(np.max(a_mag))


def run_inference(ax, ay, az, gx, gy, gz):
    """Full inference pipeline: feature extraction -> scale -> predict."""
    feat, peak_mag = extract_features(ax, ay, az, gx)

    # Build feature vector in the exact order the model was trained on
    X = np.array([[feat[name] for name in FEATURE_NAMES]], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_scaled = _scaler.transform(X)
    label    = int(_model.predict(X_scaled)[0])
    proba    = _model.predict_proba(X_scaled)[0]
    confidence = float(np.max(proba))

    # When the model is uncertain, its class boundaries are known to overlap
    # (see ML_CLASSIFICATION_ANALYSIS_DETAILED.md issue #2) — fall back to the
    # physics-based peak-magnitude thresholds instead of trusting a low-confidence label.
    label_source = "model"
    if confidence < CONFIDENCE_THRESHOLD:
        label_source = "physics_fallback"
        if peak_mag < ACCIDENT_G_THRESHOLD:
            label = 0
        elif peak_mag < SEVERE_G_THRESHOLD:
            label = 1
        else:
            label = 2

    # Dual-validation: model must predict non-normal AND physics threshold confirms.
    # This prevents model noise from generating false accident alerts.
    accident_confirmed = bool(label >= 1 and peak_mag >= ACCIDENT_G_THRESHOLD)

    return {
        "severity_class":    label,
        "severity_name":     CLASS_NAMES[label],
        "confidence":        round(confidence, 4),
        "label_source":      label_source,
        "probabilities": {
            "Normal":   round(float(proba[0]), 4),
            "Moderate": round(float(proba[1]), 4),
            "Severe":   round(float(proba[2]), 4),
        },
        "accident_confirmed": accident_confirmed,
        "peak_magnitude_g":  round(peak_mag, 4),
        "features":          {k: round(v, 6) for k, v in feat.items()},
    }


# ── Validation helpers ─────────────────────────────────────────────────────────
def _parse_imu_field(data, field):
    """
    Extract a numeric 1-D array from a JSON field.
    Returns (array, error_string_or_None).
    """
    if field not in data:
        return None, f"Missing required field: '{field}'"

    try:
        arr = np.array(data[field], dtype=float)
    except (TypeError, ValueError) as exc:
        return None, f"Field '{field}' must be a numeric array: {exc}"

    if arr.ndim != 1:
        return None, f"Field '{field}' must be a 1-D array, got shape {arr.shape}"

    if len(arr) != WINDOW_SAMPLES:
        return None, (
            f"Field '{field}' must have exactly {WINDOW_SAMPLES} samples, "
            f"got {len(arr)}"
        )

    if not np.all(np.isfinite(arr)):
        return None, f"Field '{field}' contains NaN or Inf values"

    return arr, None


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Liveness / readiness probe."""
    return jsonify({
        "status":     "healthy",
        "model":      "xgboost_no_proxy",
        "n_features": len(FEATURE_NAMES),
        "window_samples": WINDOW_SAMPLES,
        "sample_rate_hz": TARGET_FS,
        "classes":    CLASS_NAMES,
        "accident_threshold_g": ACCIDENT_G_THRESHOLD,
        "severe_threshold_g":   SEVERE_G_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify a 5-second IMU window as Normal / Moderate / Severe.

    Request JSON:
        {
            "ax": [float, ...],   # 500 samples, g-units
            "ay": [...],
            "az": [...],
            "gx": [...],          # deg/s or rad/s (not used in features directly
            "gy": [...],          #   but required for completeness / future models)
            "gz": [...]
        }

    Response JSON:
        {
            "severity_class":     0 | 1 | 2,
            "severity_name":      "Normal" | "Moderate" | "Severe",
            "confidence":         float,
            "label_source":       "model" | "physics_fallback",
            "probabilities":      {"Normal": float, "Moderate": float, "Severe": float},
            "accident_confirmed": bool,     # model AND physics both agree
            "peak_magnitude_g":   float,
            "features":           {name: float, ...},
            "inference_time_ms":  float
        }
    """
    t0 = time.perf_counter()

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or empty JSON body"}), 400

    # Parse and validate all six channels
    arrays = {}
    for field in ("ax", "ay", "az", "gx", "gy", "gz"):
        arr, err = _parse_imu_field(data, field)
        if err:
            return jsonify({"error": err}), 400
        arrays[field] = arr

    try:
        result = run_inference(
            arrays["ax"], arrays["ay"], arrays["az"],
            arrays["gx"], arrays["gy"], arrays["gz"],
        )
    except Exception as exc:
        log.exception("Inference error")
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    result["inference_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return jsonify(result), 200


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
