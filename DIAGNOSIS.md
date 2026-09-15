# DIAGNOSIS — every event classified "Normal"

Date: 2026-09-14 · Repo `bernardinee/accident-severity-api` · Branch `fix/all-normal-diagnosis` (off `main` @ `6530ee1`)
Raw evidence (scripts, full response bodies, server logs): `OneDrive\Desktop\cloud_api_diagnosis_20260914\`

---

## 1. Root cause

**The API is correct. Neither input source reaching it contains a valid crash window:
(a) the ESP32 firmware sent the 500-sample window the instant it triggered, so the crash pulse was ~10 ms at the window edge; and
(b) the v1 Postman collections mix units (gravity in m/s², pulses in g), so the unit guard divides every pulse by 9.81.**

Two input paths were in use and both produce all-Normal. Neither involves the model or the decision logic.

### 1a. Device path: capture window ends at the trigger sample

`ESP32_Complete_3LED_System.ino` (the sketch posting to production) sets `crash_detected_flag`, stores the trigger sample, then calls `processCrash()` in the **same `loop()` pass**. The window posted is the 499 samples *before* the trigger plus the trigger sample. The pulse starts there, so the window holds almost none of it.

The task brief says "the firmware then begins collecting 5 s". The code does the opposite: it collects nothing after the trigger. The effect is the same, but the evidence below looks different from what that hypothesis predicts.

Evidence: 200 verified VZCrash crash recordings replayed through the firmware's exact trigger (|a| ≥ 2 g and |jerk| ≥ 5 g/s at 100 Hz) and through `run_inference()` from `main`:

| Window | Normal | Moderate | Severe | `label_source` | median `p_crash` | median `excursion_ms` |
|---|---|---|---|---|---|---|
| as built (ends at trigger) | **200** | 0 | 0 | `model` ×200 | **0.0001** | **10** |
| impact-centred (250 pre + 250 post) | 39 | 144 | 17 | `model+signature` ×161 | 0.7724 | 60 |

Trigger peak (what the firmware prints) vs `peak_g` the API returns for the same event:

```
recording     firmware trigger | API as-built                          || API centred
vzc_1764081   2.04 g           | 2.04 g  10 ms Normal (model, p=0.0000) ||  3.74 g 100 ms Moderate (model+signature, p=0.8605)
vzc_1917456   2.04 g           | 2.04 g  10 ms Normal (model, p=0.0001) ||  4.98 g 110 ms Severe   (model+signature, p=0.8988)
vzc_1610453   2.16 g           | 2.16 g  10 ms Normal (model, p=0.0001) ||  4.67 g 150 ms Severe   (model+signature, p=0.7367)
vzc_1643833   3.21 g           | 3.21 g  10 ms Normal (model, p=0.0004) ||  5.92 g 210 ms Severe   (model+signature, p=0.6587)
vzc_0751408   2.70 g           | 2.70 g  10 ms Normal (model, p=0.0005) ||  2.79 g  40 ms Moderate (model+signature, p=0.5576)
median (n=200) 2.22 g          | 2.23 g                                 ||  3.39 g
```

The numbers agree, because the trigger sample is the window's last sample. What differs is the excursion (10 ms, and the model has never seen the pulse) and the real peak (3.39 g), which comes *after* the trigger. `unit_scale_applied` = 1.0 for all 200, so the device's units are fine.

Full response bodies, same recording (`vzc_1764081`):

```json
as built: {"severity_class": 0, "severity_name": "Normal", "confidence": 1.0, "p_crash": 0.0, "model_severity": "Normal", "accident_confirmed": false, "probabilities": {"Normal": 1.0, "Moderate": 0.0, "Severe": 0.0}, "crash_signature": {"peak_g": 2.045, "excursion_ms": 10.0, "impulse_gs": 0.01, "signature_match": false}, "unit_scale_applied": 1.0, "label_source": "model"}
centred:  {"severity_class": 1, "severity_name": "Moderate", "confidence": 0.8605, "p_crash": 0.8605, "model_severity": "Moderate", "accident_confirmed": true, "probabilities": {"Normal": 0.1395, "Moderate": 0.8605, "Severe": 0.0}, "crash_signature": {"peak_g": 3.742, "excursion_ms": 100.0, "impulse_gs": 0.493, "signature_match": true}, "unit_scale_applied": 1.0, "label_source": "model+signature"}
```

> Limitation: this is a software replay of the firmware's capture logic on verified crash data. There is no serial log from the physical unit. To confirm on hardware, flash the fixed sketch; it now prints `P(crash)`, pulse length and `label_source` for each event.

Why it only surfaced after the v2 deploy: the v1 API had no duration gate and classified on peak height, so a 2 g sample at the window edge was enough. v2 requires the pulse itself.

A second firmware defect makes this worse: `mpu.initialize()` leaves the MPU6050 at **±2 g**, clipping each axis (max resultant 3.46 g). With an impact-centred window, clipping lowers crash recall from 80.5% to 73%. It does not by itself cause all-Normal.

### 1b. Postman path: mixed units in the v1 collections

`generate_postman_scenarios_DEBUG.py` builds `az = np.random.normal(9.8, 0.2, 500)` (gravity in m/s²) and adds pulses such as `peak_g=22.0` in g. Median |a| ≈ 9.8 > 5, so `normalize_to_g()` divides **all** channels by 9.80665, and the "22 g" pulse arrives as 2.55 g.

Live responses, every scenario (`responses_live.jsonl`):

| Collection | Result |
|---|---|
| `Postman_Tests_FINAL.json` | 19/19 → HTTP 200, `unit_scale_applied: 9.80665`, `Normal` |
| `Postman_Tests_DEBUG.json` | 19/19 → HTTP 200, `unit_scale_applied: 9.80665`, `Normal` |
| `Postman_Tests_Fixed.json` | 19/19 → HTTP 400 `Missing field: gx` (body has only ax/ay/az) |
| `Postman_Accident_Detection_Tests.json` | 21/21 → HTTP 400 `Missing field: gx` (same) |

The two 400 collections were never valid: both v1 commits (`fb5cf15`, `72da354`) also required all six fields.

The same FINAL windows with `az` rebased to 1 g (units corrected, nothing else changed) give the v2 taxonomy's answers. Pothole 4.5 g/80 ms → **Moderate**; speed bump 5.2 g/160 ms → **Moderate**; all ≥7 g "Severe" scenarios → Normal via `signature_override` (correct per v2); 520 ms/310 ms "manoeuvres" → Normal (excursion > 250 ms). See `checks_output.txt`.

---

## 2. Investigation record and what was ruled out

### Step 1: which model is live? → **main, yes**

```
GET /health  →  HTTP/1.1 200 OK   (2026-09-14 21:49:31 GMT, server railway-hikari)
{"classes":["Normal","Moderate","Severe"],"crash_alert_threshold":0.396,"model":"phase2_xgboost_calibrated","n_features":25,"status":"healthy","taxonomy":"signature+impulse (v2)"}
```

Deployment history (GitHub deployment/commit statuses posted by Railway, public API):

| Commit | Event | Railway status |
|---|---|---|
| `8c91905` merge PR #1 (v2) | 2026-08-18 23:48:14Z | **failure** "Deployment failed" |
| `6530ee1` "Trigger Railway rebuild" (empty commit, = `main` HEAD) | in_progress 11:17:32Z → **success 2026-08-19 11:18:50Z** | |
| `72da354` (July, v1) deployment | **inactive** 2026-08-19 11:19:03Z | |

The failed build was superseded 12 hours later by a successful build of identical content. Response bodies from live and from a local run of `git archive HEAD` are **identical on all 44 successful Postman scenarios** (ignoring `inference_time_ms`), and every 400 error matches.

**The live endpoint is running the code currently on `main`: YES.** The "stale Phase 1 model" hypothesis is ruled out.

### Step 2: bisect → **the API is fine; the data is the problem**

Both targets classify the known-good crash scenarios correctly (`smoke_test.py` and `Postman_v2_Tests.json`, identical on local and live):

```
[PASS] normal driving             -> Normal    p_crash=0.00 sig=False
[PASS] brief spike (drop) 12 g    -> Normal    p_crash=0.49 sig=False
[PASS] sustained 9 g maneuver     -> Normal    p_crash=0.00 sig=False
[PASS] real crash 4 g / 90 ms     -> Moderate  p_crash=0.98 sig=True
[PASS] real crash 6 g / 150 ms    -> Moderate  p_crash=0.81 sig=True
[PASS(known-miss)] MISSED-19% crash 3 g/30 ms -> Normal p_crash=0.60 sig=False
```

`test_api.py`: **5/8 on both local and live**, identical bodies. Its failures do not indicate a broken API:

| Test | Response | Why |
|---|---|---|
| 3 "Moderate", 3.5 g Gaussian σ=150 ms | Normal, `signature_override`, p_crash 0.6724, excursion **360 ms** | > 250 ms: a sustained event, not an impact (v2 by design) |
| 4 "Severe", 9 g pulse | Normal, `signature_override`, p_crash 0.4569, peak **9.05 g** | ≥ 7 g band removed by design; this test encodes the retired peak≥7 g rule |
| 5d NaN in `ax` | **HTTP 200, Normal** (v1 prod run in `api_test_results_prod.txt`: HTTP 400) | **Real regression**; see §3 |

Comparison with `api_test_results_prod.txt` (2026-05-14): that run hit `xgboost_no_proxy` (v1), where the 3.7 g and 9.0 g windows were Moderate/Severe by peak height. The class differences are the v2 taxonomy change; the 5d difference is a defect.

### Step 3: API checks (run anyway, against real crash windows) → all ruled out

| Check | Evidence | Verdict |
|---|---|---|
| a. Threshold | `task6_calibration.json` loaded: server log `Model loaded (25 features); crash-alert P(crash) threshold=0.396`; `/health` 0.396 (fallback would be 0.5). The key `crash_alert_high_precision_0p90` holds **0.396**, not 0.8–0.9, and it is *lower* than `crash_alert_f1_optimal` 0.437. | Not a cause |
| b. Filter attenuation | 200 centred crash windows: raw peak median 3.482 g → filtered 3.389 g (**1.45%** median); 3/200 dropped from ≥2 g to <2 g | Not a cause |
| c. Excursion | filtered excursion p25/median/p75 = **40 / 60 / 110 ms** (raw median 70 ms). Gate failures: exc<40 ms 28, exc>250 ms 3, peak<2 g 3, peak≥7 g 0. That is the documented ~81% recall ceiling, not a defect. | Not a cause |
| d. Feature order | `n_features_in_` 25; `phase2_feature_names.json` == `features_25()` keys == training `p3_common.FEATURES_25` order. Model is `CalibratedClassifierCV` over `Pipeline(['sc','clf'])`, so the scaler is inside the pickle and serving raw features is correct. | Not a cause |
| e. Artifact | `phase2_xgboost_calibrated.joblib` sha256 `2f5d9714…` = `phase3/deploy` (20 Hz retrain) ≠ `phase2/deploy` (`5543d13d…`); added once in `8ed9d44` | Not a cause |
| Units | `unit_scale_applied` 1.0 on all device-replay windows | Not a cause for device path; **is** the cause for v1 Postman (§1b) |

### Other defects found (not causing all-Normal; not changed)

1. **Unit guard misfires on long high-g windows.** `normalize_to_g` uses median |a| > 5, so a window with >50% of samples above 5 g is treated as m/s². Seen on `Postman_v2_Tests` "Sustained 9 g maneuver": `unit_scale_applied: 9.80665`. That scenario still returns Normal (as expected), but for the wrong reason. Real crashes (~70 ms) cannot trigger it.
2. **Threshold provenance.** `task6_calibration.json` on `main` is byte-identical to the Phase 2 file (sha `ce353f32…`). The 0.396 was derived for the Phase 2 model, not re-derived for the deployed 20 Hz retrain. Its `0p90` / `f1_optimal` names are also inverted.
3. **Model hold-out F1.** The deployed artifact's hold-out macro-F1 is **0.9439** (`phase3/outputs/task5_retrain.json`), not 0.948 (the pre-retrain model).
4. **Stale tests.** `test_api.py` tests 3–4 and the v1 Postman collections encode the retired peak-height taxonomy, and two collections omit gyro fields. Anyone testing with them sees "all Normal".
5. **`app.py` docstring** still says "Phase 2 (STAGED, not deployed)" although it is live.

---

## 3. Fixes applied

### 3a. Firmware (resolves the device-path root cause)

`OneDrive\Desktop\ESP32_Complete_3LED_System\ESP32_Complete_3LED_System.ino`. This is not in this repo or any git repo; backup at `.ino.BACKUP_20260914`. Full diff: `firmware.diff`.

- After the trigger, keep sampling `POST_TRIGGER_SAMPLES = 250`, then send. The window is 250 pre + 250 post, impact centred; still 500 samples at 100 Hz, so the contract is unchanged.
- `mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_16)` with `RAW_ACCEL_TO_G = 1/2048` (was the ±2 g default with 1/16384).
- Removed the 600 ms blocking `delay()` flash in the trigger path; it would have dropped the post-impact samples.
- Send no longer requires `wifi_ready` (a trigger with WiFi down left detection stuck forever); it falls through to the existing red "API unavailable" alert.
- Results box reads `crash_signature.peak_g` (v2 has no `peak_magnitude_g`) and prints `p_crash`, `excursion_ms`, `label_source`.

Compiles: `arduino-cli compile --fqbn esp32:esp32:esp32` → `Sketch uses 1090675 bytes (83%)`.

Before/after, same window, same code: see the tables in §1a. As-built 0/200 crashes detected, fixed capture 161/200 (80.5%).

### 3b. API (commit on this branch): restore NaN/Inf rejection

v2 dropped the check both v1 commits had. A `null` from a device (ArduinoJson serialises NaN as `null`) becomes NaN, `nan_to_num` zeroes the features, and the request silently returns Normal:

```
before: POST /predict with ax[0]=NaN → HTTP 200 {"severity_name": "Normal", "p_crash": 0.0, "crash_signature": {"peak_g": NaN, ...}, "label_source": "model"}
after:  POST /predict with ax[0]=NaN → HTTP 400 {"error": "Field 'ax' contains NaN or Inf values"}
```

Four lines in `predict()`; this restores the v1 contract (`api_test_results_prod.txt` 5d), and valid requests are unaffected (§4). No model, threshold, gate or response-field change.

**Not fixed, deliberately:** the v1 Postman collections and `test_api.py` tests 3–4. Correcting them means rewriting expectations to the v2 taxonomy, which is a test redesign, not a minimal fix. Use `smoke_test.py` and `Postman_v2_Tests.json` (untracked in the working tree) as the valid v2 suite.

---

## 4. Verification (patched build on :5058 vs unpatched `main` on :5057)

| Suite | Unpatched `main` (local = live) | Patched |
|---|---|---|
| `smoke_test.py` | 6/6 | 6/6, identical output |
| `test_api.py` | 5/8 (3, 4, 5d fail) | **6/8** (5d now PASS: `400 Field 'ax' contains NaN or Inf values`); 3–4 unchanged, v1 expectations |
| `Postman_v2_Tests.json` (8) | 6 × 200 + 2 × 400, crashes → Moderate | identical |
| `Postman_Tests_FINAL.json` (19) | 19 × Normal, scale 9.80665 | identical |
| `Postman_Tests_DEBUG.json` (19) | 19 × Normal, scale 9.80665 | identical |
| `Postman_Tests_Fixed.json` (19) | 19 × 400 missing gx | identical |
| `Postman_Accident_Detection_Tests.json` (21) | 21 × 400 missing gx | identical |
| **All 86 Postman requests** | | **44/44 200-bodies byte-identical, 42/42 400s identical, 0 differences** |

Scenario-by-scenario tables: `postman_comparison.txt` (local vs live) and `postman_patched_vs_main.txt` (patched vs main). Full bodies: `responses_*.jsonl`.

---

## 5. Deployment status

- **Live** = `main` @ `6530ee1`, Railway deploy success 2026-08-19 11:18:50Z. There is no failing build now; the only failure (`8c91905`, 2026-08-18 23:48Z) was superseded.
- **The build log for that failure could not be retrieved.** The Railway CLI 4.58.0 on this machine returns `Unauthorized. Please login with 'railway login'`, and GitHub carries only the status text "Deployment failed". To pull it: `! railway login`, then `railway logs --build` for that deployment in project `7e4b7f88-37d7-4aee-beb2-29c76d52a6a5`. Since the identical tree built successfully on retry, the cause was not in the code.
- **The API fix (§3b) is NOT live.** It is committed on `fix/all-normal-diagnosis`, not pushed. Railway deploys `main`, so it goes live only after push + merge.
- **The firmware fix (§3a) is on disk only.** It must be flashed to the ESP32. That is the fix that resolves the reported symptom, and it needs no API deploy.

---

## 6. Phase 5 follow-up (2026-09-15): fixed, fine-tuned, redeployed

The §5 status above is superseded. Full method and evidence: `docs/PHASE5_REPORT.md`.

**Fixed**
- **Severity grade biased against Severe:** argmax → calibrated P(Severe | crash) ≥ 0.66.
- **Train/serve skew:** retrained on the features this service computes.
- **Stale threshold:** re-selected on group-aware out-of-fold predictions.
- **Calibration:** group-aware isotonic calibration; test ECE 0.037 → 0.0023.
- **Unit guard:** uses the 10th percentile, so long high-g windows are no longer rescaled.
- **NaN/Inf/null:** rejected with 400.
- **Test suite:** designed Normal/Moderate/Severe cases, random real hold-out windows, contract cases; v1 assets moved to `legacy/`.
- **Firmware** (outside this repo): impact-centred capture, ±16 g, ±500 dps, boot mount-orientation check.

**Held-out system results** (48,989 windows, 10,806 recordings never used in training or tuning)

| | Before | After |
|---|---|---|
| Macro-F1 | 0.9579 [0.9529, 0.9625] | **0.9641 [0.9594, 0.9684]** (Δ +0.0062 [+0.0024, +0.0098], p = 0.001) |
| Recall N / M / S | 0.994 / 0.947 / 0.892 | 0.995 / 0.937 / **0.952** |
| Normal false-alarm rate | 0.58 % | 0.54 % |

Deployment and live verification results are recorded in `docs/PHASE5_REPORT.md` §7.
