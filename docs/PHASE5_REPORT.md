# Phase 5 — Fix, fine-tune and redeploy the crash-severity system

Date: 2026-09-15 · Follows `cloud_api/DIAGNOSIS.md` (2026-09-14)
Code: `phase5/*.py` · Results: `phase5/results/` · Deployed artifacts: `cloud_api/artifacts/`

---

## 1. Summary

| | Before (deployed 2026-08-19) | After (Phase 5) |
|---|---|---|
| ESP32 firmware capture | window **ended at the trigger**, so crashes were never inside it | impact-centred 2.5 s + 2.5 s, ±16 g, ±500 dps, mount check |
| Triggered crash recordings alerted (device replay, 399 test-group recordings) | **1** | **312** (of 30 true-Severe events: 26 Severe, 3 Moderate, 1 Normal) |
| API system macro-F1, group hold-out (48,989 windows) | 0.9579 [0.9529, 0.9625] | **0.9641 [0.9594, 0.9684]** |
| Severe recall | 0.892 [0.874, 0.911] | **0.952 [0.938, 0.965]** |
| Normal false-alarm rate | 0.58 % | **0.54 %** |
| P(crash) calibration error (ECE) | 0.037 | **0.0023** |
| Unit guard on long high-g windows | misfired (rescaled by 1/9.81) | fixed |
| NaN / Inf / null samples | HTTP 200 "Normal" | HTTP 400 |
| Test suite | v1 expectations, mixed units, no Severe case | 138 checks incl. designed Normal/Moderate/Severe + random real hold-out windows |

All three classes are produced, by the model, on real and designed inputs. No peak-g rule was introduced; the v2 taxonomy (2–7 g, 40–250 ms signature, impulse-graded) is unchanged.

---

## 2. What was wrong (evidence)

1. **Firmware (root cause of "everything Normal").** Replaying 400 random test-group crash recordings through the as-built capture, 398/399 triggered events returned Normal with the old *and* the new API (`results/firmware_replay.json`). See DIAGNOSIS.md §1a.
2. **Severity grade biased against Severe.** The API graded by argmax. Severe is 7× rarer than Moderate, so 110 of 1,216 held-out Severe windows came back Moderate (Severe recall 0.892 vs Moderate 0.947).
3. **Train/serve skew.** The deployed model scored 0.9439 on its training-path features but **0.9360** on the features the API actually computes. The API re-applies the 20 Hz filter to already-filtered training signals.
4. **Threshold provenance.** The 0.396 threshold was derived for the Phase 2 model, not the deployed retrain, and under a mislabelled key.
5. **Calibration.** The deployed isotonic calibration used non-group-aware `cv=3`; test ECE 0.037.
6. **Unit guard misfire.** Median |a| > 5 treated g windows with more than 2.5 s above 5 g as m/s².
7. **NaN regression.** The v2 rewrite dropped v1's finite-value check.
8. **Firmware gyro range.** ±250 dps is exceeded by 2.3 % of verified crashes (0.6 % exceed ±500).

---

## 3. Method — how bias was kept out

**Serve-path store (`p5_build_store.py`).** All 247,711 windows of `combined_20hz.csv` were rebuilt from raw signals: VZCrash from the 4.3 GB source CSV, Phase-1 re-processed at 20 Hz. Each was pushed through the exact `app.py` functions: unit guard, 20 Hz filter, `features_25`, `crash_signature`.
- **Alignment:** training-path features recomputed from the same signals match the stored ones to max relative error 6.3e-16 on every row.
- **Gate consistency:** on the served signal the signature gate passes 98.9 % of true Moderate and 98.6 % of true Severe test windows. This is the system recall ceiling: 112 of 9,892 crash windows cannot pass.

**Protocol (`p5_tune.py`, `p5_finalize.py`).**
- **TEST** = the identical group-disjoint 20 % hold-out the deployed model was scored on: `GroupShuffleSplit(test_size=0.2, random_state=42)`, 10,806 recordings, 0 groups shared with training. Reproduction check: the deployed model scores 0.9439 on it, matching `task5_retrain.json`.
- **Every choice was made on 5-fold `StratifiedGroupKFold` out-of-fold predictions inside the 80 % training groups:**
  - hyperparameters: 10 configurations × 3 class-weighting schemes (`hyperparameter_search.csv`);
  - isotonic P(crash) calibration, cross-fitted OOF ECE 0.0008;
  - crash threshold and Severe threshold, via a grid of 4,753 pairs.
- **The test set was scored once per final system.**
- **Operating-point policy, fixed in code before the tuned system was scored on test:** maximise macro-F1 (Normal, Moderate and Severe weighted equally) subject to Normal false-alarm rate ≤ 0.58 %, the deployed system's rate. Gains may not be bought with extra ambulance call-outs.
- **Uncertainty:** 1,000–2,000× group bootstrap CIs, a paired group bootstrap, and McNemar's test.
- **Severe grading stays a model decision:** P(Severe | crash) ≥ threshold. Grading by the impulse formula would copy the label definition — the circularity the thesis removed — so it is excluded.

**Selected model:** XGBoost on serve-path features; `n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.7, colsample_bytree=1.0, min_child_weight=1, reg_lambda=5`; balanced class weights; isotonic P(crash) calibration.
- **Stability:** the top 10 of 30 search candidates lie within 0.0006 OOF macro-F1.
- **Operating point:** crash threshold **0.42**; Severe threshold **P(Severe | crash) ≥ 0.66**.

---

## 4. Results on the group-disjoint hold-out (48,989 windows)

| System | Macro-F1 [95 % CI] | Normal R | Moderate R | Severe R | Severe P | False alarm | Crash recall |
|---|---|---|---|---|---|---|---|
| A. Deployed (Aug-19) | 0.9579 [0.9529, 0.9625] | 0.9942 | 0.9465 | 0.8923 | 0.9670 | 0.58 % | 0.9530 |
| E. New model, old rule | 0.9642 [0.9596, 0.9685] | 0.9940 | 0.9441 | 0.9605 | 0.9270 | 0.60 % | 0.9562 |
| **D. Deployed now** | **0.9641 [0.9594, 0.9684]** | **0.9946** | 0.9365 | **0.9515** | 0.9437 | **0.54 %** | 0.9473 |
| C. No false-alarm cap | 0.9687 [0.9632, 0.9735] | 0.9914 | 0.9773 | 0.9531 | 0.9339 | 0.86 % | 0.9833 |

Per-class F1 for D: Normal 0.9907, Moderate 0.9540, Severe 0.9476 (A: 0.9912, 0.9544, 0.9281).

Confusion matrices (rows true N/M/S, columns predicted):

```
A deployed   [[38870, 210, 17], [444, 8212, 20], [21, 110, 1085]]
D now        [[38885, 192, 20], [502, 8125, 49], [19,  40, 1157]]
```

**Significance, D vs A:** paired group bootstrap Δ macro-F1 = **+0.0062, 95 % CI [+0.0024, +0.0098], p = 0.001**. McNemar on row accuracy: 157 vs 157 discordant, p = 1.0. Overall accuracy is unchanged (0.9832); the gain is in **balance across classes**, which is what macro-F1 measures.

**Trade-off, stated plainly.** Under the false-alarm cap, Moderate recall falls 1.0 point: 502 vs 444 Moderate windows returned Normal. System C removes that loss (Moderate recall 0.977, crash recall 0.983) at +0.28 points of false alarms. That is significantly better macro-F1 than D (Δ +0.0046 [+0.0026, +0.0064]) but breaks the policy. **To switch to C:** set `crash_threshold` to 0.09 in `cloud_api/artifacts/decision_config.json`. Nothing else changes, because the calibration and Severe threshold are shared.

**By source (D).**
- **VZCrash windows** (the device domain: gravity on Z, gyro present): macro-F1 0.9646 (A 0.9583), Severe recall 0.9515 (A 0.8923), false alarms 0.64 % (A 0.68 %).
- **Phase-1 non-crash motion:** false alarms 0.12 % (A 0.14 %).

**Calibration (test):** ECE of P(crash) 0.0023 (deployed 0.037; new model uncalibrated 0.026).

---

## 5. Device-level replay (`p5_firmware_replay.py`)

400 random test-group VZCrash crash recordings and 400 normal/near-miss recordings were streamed through the firmware trigger (2 g and 5 g/s jerk). Truth = v2 label of the impact-centred window.

| Firmware | API | True Severe (30) | True Moderate (297) | Normal drives alerted (of 400) |
|---|---|---|---|---|
| as built | old | 0 Severe, 1 Moderate | 0 | 0 |
| as built | new | 0 Severe, 1 Moderate | 0 | 0 |
| fixed | old | 22 Severe, 7 Moderate, 1 Normal | 282 | 5 |
| **fixed** | **new** | **26 Severe**, 3 Moderate, 1 Normal | **282** | **3** |

Exact grade agreement on signature-crash events (327): old API 0.930, new API **0.942**. True Moderate events: 15 returned Normal under both APIs. Of the 72 triggered crash recordings whose centred window carries no crash signature (truth Normal by the taxonomy), 71 return Normal and 1 Moderate under both APIs.

---

## 6. Changes made

**API — `cloud_api`**
- `app.py`: config-driven model and operating point (`artifacts/decision_config.json`, no silent fallback); isotonic P(crash) calibration; Severe = P(Severe | crash) ≥ 0.66; unit guard uses the 10th percentile; NaN/Inf/null → 400; `/health` reports `severe_ratio_threshold`. `/predict` request and response fields are unchanged.
- Artifacts: `p5_xgboost.joblib` (sha256 `d5af31c1…`), `p5_feature_names.json`, `decision_config.json`. Previous `phase2_*` kept for rollback.
- Tests:
  - `tests/make_fixtures.py` → `tests/fixtures.json` and `Postman_v2_Tests.json`: 7 designed cases incl. Severe and an m/s² input, 18 random real hold-out windows, 6 contract cases.
  - `test_api.py` rewritten: 138 checks. `smoke_test.py`: the 7 designed cases.
  - v1 assets moved to `legacy/` with a README.

**Firmware — `ESP32_Complete_3LED_System.ino`** (backup `.ino.BACKUP_20260914`; compiles, 1,091,483 bytes)
- Impact-centred capture (250 pre + 250 post), ±16 g (1/2048 g per LSB).
- **±500 dps gyro** (1/65.5 dps per LSB).
- No blocking delay in the trigger path; send without WiFi falls back to the red alert.
- **Boot-time mount check**: warns if gravity is not on +Z, because the model's per-axis features assume it.
- Serial prints P(crash), pulse length and label_source.

---

## 7. Verification

**Local** (working tree, `python app.py`):
- `smoke_test.py` 7/7;
- `test_api.py` **138 passed, 0 failed**;
- API vs offline pipeline parity on the 18 real windows: 18/18 (class, signature, p_crash within 1e-4);
- clean virtualenv from `requirements.txt` reproduces the fixtures.

**Production:** see `cloud_api/DIAGNOSIS.md` §6 (deployment log, live suite results).

---

## 8. Limitations (unchanged by this phase, stated for the thesis)

- **Severity has no annotated ground truth.** Moderate vs Severe is the impulse proxy (85th percentile of VZCrash impulse ≈ 34 km/h delta-v).
- **System recall ceilings.**
  - 10.8 % of VZCrash crashes were removed at ingest (<2 g);
  - the 2 g firmware trigger;
  - the 40 ms duration floor (the labels define a crash as ≥ 40 ms, so the floor was not moved);
  - 1.1 % of crash windows fail the gate on the served signal.
- **Phase-1 non-crash data is gravity-removed**, and 72 % lacks gyro, unlike the device. VZCrash normal/near-miss recordings (158,848 windows, gravity on Z and gyro present) supply the device-domain Normal class.
- **No labelled recordings from the physical prototype exist.** Everything above is on public data and software replay. The next step is a bench and vehicle validation with the mount check passing.

---

## 9. Thesis numbers that change

Chapter 3/4 quote the deployed model's hold-out macro-F1 **0.9439**. That figure is model-only (argmax, no gate) on training-path features. On the same hold-out:

| Figure | Deployed (Phase 3) | Phase 5 |
|---|---|---|
| Model only, argmax, training-path features | 0.9439 | – (Phase 5 is trained on serve-path features) |
| Model only, argmax, serve-path features (what the API computes) | 0.9360 | not a deployment metric: balanced class weights make argmax over-call crashes by design; thresholds replace argmax |
| **Full system as served** (model + gate + operating point) | **0.9579 [0.9529, 0.9625]** | **0.9641 [0.9594, 0.9684]** |

Report the full-system figure together with the operating-point policy (§3). It is the number that describes what `/predict` returns. Do not compare it directly with 0.9439.
