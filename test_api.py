"""
Test suite for the Accident Severity Classification API.

Generates synthetic IMU windows for all three severity classes and verifies:
  - Health check endpoint
  - Class 0 (Normal) prediction
  - Class 1 (Moderate) prediction
  - Class 2 (Severe)  prediction
  - Error handling (missing fields, wrong array length, non-JSON body)

Saves full results to api_test_results.txt.

Usage:
    Start the API first:  python app.py
    Then run:             python test_api.py
"""

import json
import time
import urllib.error
import urllib.request

import numpy as np

BASE_URL     = "http://localhost:5000"
RESULTS_FILE = "api_test_results.txt"
PROD_URL     = "https://accident-severity-api-production.up.railway.app"
FS           = 100
N            = 500   # samples per window (5 s @ 100 Hz)

# ── Synthetic window generators ────────────────────────────────────────────────
# All values are in g-units (gravity = 1g).
# The model was trained on g-unit data (preprocess.py converts m/s² → g).
# The deployment sensor (ESP32 + MPU6050) should send data in g-units.

def _gaussian_pulse(n, center, sigma, amplitude):
    t = np.arange(n)
    return amplitude * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))


def make_normal_window(seed=42):
    """
    Class 0: smooth driving.
    az ≈ 1g (gravity on Z-axis, dashboard-flat mount).
    peak a_mag ≈ 1.0 g  →  below 2g threshold.
    """
    rng = np.random.default_rng(seed)
    ax = rng.normal(0.00, 0.05, N)
    ay = rng.normal(0.00, 0.05, N)
    az = rng.normal(1.00, 0.02, N)   # ~1g gravity component
    gx = rng.normal(0.00, 0.50, N)
    gy = rng.normal(0.00, 0.50, N)
    gz = rng.normal(0.00, 0.50, N)
    return ax, ay, az, gx, gy, gz


def make_moderate_window(seed=43):
    """
    Class 1: harsh braking / pothole event.
    Injects a 3.5g Gaussian pulse on ay (longitudinal axis).
    peak a_mag ≈ sqrt(0^2 + 3.5^2 + 1^2) ≈ 3.6 g  →  in [2g, 7g] range.
    """
    rng = np.random.default_rng(seed)
    ax = rng.normal(0.00, 0.05, N)
    ay = rng.normal(0.00, 0.05, N)
    az = rng.normal(1.00, 0.02, N)
    gx = rng.normal(0.00, 0.50, N)
    gy = rng.normal(0.00, 0.50, N)
    gz = rng.normal(0.00, 0.50, N)
    ay += _gaussian_pulse(N, center=200, sigma=15, amplitude=3.5)
    return ax, ay, az, gx, gy, gz


def make_severe_window(seed=44):
    """
    Class 2: severe crash impact.
    Injects a 9g Gaussian pulse on ay (frontal collision).
    peak a_mag ≈ sqrt(0^2 + 9^2 + 1^2) ≈ 9.1 g  →  above 7g threshold.
    """
    rng = np.random.default_rng(seed)
    ax = rng.normal(0.00, 0.05, N)
    ay = rng.normal(0.00, 0.05, N)
    az = rng.normal(1.00, 0.02, N)
    gx = rng.normal(0.00, 0.50, N)
    gy = rng.normal(0.00, 0.50, N)
    gz = rng.normal(0.00, 0.50, N)
    ay += _gaussian_pulse(N, center=250, sigma=8, amplitude=9.0)
    return ax, ay, az, gx, gy, gz


# ── HTTP helpers ───────────────────────────────────────────────────────────────
def _get(path, timeout=5):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read())


def _post(path, body, timeout=15):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read())


def _post_raw(path, data, headers, timeout=5):
    """POST with custom data/headers for error-case tests."""
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data, headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _arrays_to_payload(ax, ay, az, gx, gy, gz):
    return {
        "ax": ax.tolist(), "ay": ay.tolist(), "az": az.tolist(),
        "gx": gx.tolist(), "gy": gy.tolist(), "gz": gz.tolist(),
    }


# ── Test runner ────────────────────────────────────────────────────────────────
def run_tests():
    lines  = []
    passed = 0
    failed = 0

    def log(msg=""):
        print(msg)
        lines.append(msg)

    def ok(msg):
        nonlocal passed
        passed += 1
        log(f"  [PASS] {msg}")

    def fail(msg):
        nonlocal failed
        failed += 1
        log(f"  [FAIL] {msg}")

    # ── Header ─────────────────────────────────────────────────────────────────
    log("=" * 65)
    log("Accident Severity API — Test Results")
    log(f"Timestamp  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Target URL : {BASE_URL}")
    log("=" * 65)

    # ── 1. Health check ────────────────────────────────────────────────────────
    log("\n[TEST 1/5]  GET /health")
    try:
        status, body = _get("/health")
        log(f"  HTTP status      : {status}")
        log(f"  model            : {body.get('model')}")
        log(f"  n_features       : {body.get('n_features')}")
        log(f"  window_samples   : {body.get('window_samples')}")
        log(f"  classes          : {body.get('classes')}")
        if status == 200 and body.get("status") == "healthy":
            ok("Health check returned 200 with status=healthy")
        else:
            fail(f"Unexpected response: {body}")
    except Exception as e:
        fail(f"Health check error: {e}")

    # ── 2–4. Severity class predictions ───────────────────────────────────────
    test_cases = [
        ("Class 0 — Normal driving",        make_normal_window,   0),
        ("Class 1 — Moderate (harsh event)", make_moderate_window, 1),
        ("Class 2 — Severe (crash impact)",  make_severe_window,   2),
    ]

    for idx, (label, factory, expected_class) in enumerate(test_cases, start=2):
        log(f"\n[TEST {idx}/5]  POST /predict — {label}")
        ax, ay, az, gx, gy, gz = factory()
        peak = float(np.max(np.sqrt(ax**2 + ay**2 + az**2)))
        log(f"  Synthetic peak a_mag : {peak:.3f} g")

        try:
            t0 = time.perf_counter()
            status, result = _post("/predict", _arrays_to_payload(ax, ay, az, gx, gy, gz))
            elapsed = (time.perf_counter() - t0) * 1000

            log(f"  HTTP status          : {status}")
            log(f"  severity_class       : {result.get('severity_class')} "
                f"({result.get('severity_name')})")
            log(f"  confidence           : {result.get('confidence'):.4f}")
            log(f"  accident_confirmed   : {result.get('accident_confirmed')}")
            log(f"  peak_magnitude_g     : {result.get('peak_magnitude_g')}")
            log(f"  probabilities        : {result.get('probabilities')}")
            log(f"  inference_time_ms    : {result.get('inference_time_ms', elapsed):.1f}")

            if result.get("severity_class") == expected_class:
                ok(f"Predicted class {expected_class} as expected")
            else:
                fail(
                    f"Expected class {expected_class}, "
                    f"got {result.get('severity_class')} "
                    f"(confidence={result.get('confidence'):.4f})"
                )
        except Exception as e:
            fail(f"Request error: {e}")

    # ── 5. Error handling ──────────────────────────────────────────────────────
    log("\n[TEST 5/5]  Error handling")

    ax, ay, az, gx, gy, gz = make_normal_window()

    # 5a — Missing field
    log("\n  5a. Missing field (no 'gz')")
    body_missing = {k: v for k, v in _arrays_to_payload(ax, ay, az, gx, gy, gz).items()
                    if k != "gz"}
    status, resp = _post_raw(
        "/predict",
        json.dumps(body_missing).encode(),
        {"Content-Type": "application/json"},
    )
    log(f"     HTTP status : {status}  error: {resp.get('error', '')}")
    if status == 400:
        ok("Missing field correctly rejected with 400")
    else:
        fail(f"Expected 400, got {status}")

    # 5b — Wrong array length
    log("\n  5b. Wrong array length (ax has 100 samples instead of 500)")
    body_short = _arrays_to_payload(ax, ay, az, gx, gy, gz)
    body_short["ax"] = ax[:100].tolist()
    status, resp = _post_raw(
        "/predict",
        json.dumps(body_short).encode(),
        {"Content-Type": "application/json"},
    )
    log(f"     HTTP status : {status}  error: {resp.get('error', '')}")
    if status == 400:
        ok("Wrong length correctly rejected with 400")
    else:
        fail(f"Expected 400, got {status}")

    # 5c — Non-JSON Content-Type
    log("\n  5c. Non-JSON Content-Type")
    status, resp = _post_raw(
        "/predict",
        b"not json data",
        {"Content-Type": "text/plain"},
    )
    log(f"     HTTP status : {status}  error: {resp.get('error', '')}")
    if status == 400:
        ok("Non-JSON Content-Type correctly rejected with 400")
    else:
        fail(f"Expected 400, got {status}")

    # 5d — Array with NaN
    log("\n  5d. Array containing NaN")
    body_nan = _arrays_to_payload(ax, ay, az, gx, gy, gz)
    body_nan["ax"] = [float("nan")] + ax[1:].tolist()
    status, resp = _post_raw(
        "/predict",
        json.dumps(body_nan).encode(),
        {"Content-Type": "application/json"},
    )
    log(f"     HTTP status : {status}  error: {resp.get('error', '')}")
    if status == 400:
        ok("NaN values correctly rejected with 400")
    else:
        fail(f"Expected 400, got {status}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total = passed + failed
    log("\n" + "=" * 65)
    log(f"Results: {passed}/{total} passed,  {failed}/{total} failed")
    log("=" * 65)

    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"\nResults saved to: {RESULTS_FILE}")
    return failed == 0


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Accident Severity API test suite")
    parser.add_argument(
        "--url",
        default=BASE_URL,
        help=f"Base URL of the API (default: {BASE_URL})",
    )
    parser.add_argument(
        "--prod", action="store_true",
        help=f"Shorthand for --url {PROD_URL}",
    )
    args = parser.parse_args()

    if args.prod:
        BASE_URL = PROD_URL
    elif args.url != BASE_URL:
        BASE_URL = args.url

    RESULTS_FILE = "api_test_results_prod.txt" if BASE_URL != "http://localhost:5000" else RESULTS_FILE

    ok = run_tests()
    sys.exit(0 if ok else 1)
