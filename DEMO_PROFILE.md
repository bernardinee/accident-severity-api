# Appendix — The `demo_rc` scaled demonstration threshold profile

> **Status of this appendix.** `demo_rc` is a *scaled demonstration configuration* for the radio-controlled (RC)
> car demonstrator. **It is not the system's operating taxonomy.** The operating taxonomy is `production`, derived
> from 27,711 verified full-size vehicle crashes (VZCrash, Phase 2). **No performance figure reported in this
> thesis derives from `demo_rc`**, and the incident store excludes every `demo_rc` event from all performance
> statistics (§A.10).
>
> **Measurement status (2026-09-16): pending.** The mechanism, collection instrumentation, derivation rules and
> analysis are implemented and tested. The RC-car dataset has not yet been collected, so §A.6–§A.9 contain no
> measured values, and `demo_rc` thresholds are `null` in `artifacts/threshold_profiles.json`. Until they are
> derived, selecting `demo_rc` on the server or in the firmware falls back to `production` and says so in every
> response and on the serial banner. No placeholder numbers were chosen, because a threshold that is not
> measured cannot be defended.

---

## A.1 Purpose and scope

The crash-signature gate that follows the classifier was derived from full-size vehicle crash pulses (Table A.1).
The physical demonstrator is an RC car, whose impacts belong to a different physical regime (§A.2). Under the
production thresholds every RC impact is correctly classified Normal, so the demonstrator cannot show the
Moderate and Severe paths working.

This appendix defines a second named threshold set, `demo_rc`, that sits beside `production` without altering it.
It documents:

1. why production is unreachable at RC scale;
2. how the profile is selected and made visible;
3. the collection protocol;
4. the pre-registered rules that turn measured RC impacts into thresholds, using the same logic that produced
   the production values;
5. the measurements, once collected;
6. the controls that keep demonstration data out of the thesis results.

**Table A.1 — Production thresholds and their provenance** (`phase2/label_v2.py`, recomputed from
`phase2/results/vzcrash_windows.csv`)

| Gate | Bound | Position in the VZCrash data |
|---|---|---|
| Peak resultant acceleration | 2.0 g ≤ peak < 7.0 g | contains 99.93% of the 27,711 crash-event peaks (0.00% below 2 g, 0.07% at or above 7 g) |
| Longest contiguous run ≥ 2 g | 40–250 ms | 17.2% of crash events fall below 40 ms, 1.8% above 250 ms (81.0% inside) |
| Severe impulse | ≥ 0.959 g·s | 85th percentile of event impulse over the 22,433 signature-matching crashes; Δv = 0.959 × 9.80665 = 9.40 m/s (33.9 km/h) |

Two clarifications matter for the derivation below.

- **The Severe cut is a percentile, not an injury boundary.** 0.959 g·s is where the 85th percentile of
  VZCrash impulse fell. That it corresponds to Δv ≈ 34 km/h, within the serious-injury Δv range, was noted as
  a physical interpretation afterwards; it was not how the number was chosen.
- **The server does not grade Severe by impulse.** Since Phase 5, `/predict` grades Severe with the model's
  P(Severe | crash) ≥ 0.66. The 0.959 g·s cut defines the training labels and grades the firmware's local
  fallback. §A.3.4 explains why `demo_rc` grades by impulse on the server.

## A.2 Why the production thresholds are unreachable at RC scale

### A.2.1 Pulse duration

An impact that brings the vehicle front to rest over crush distance *d*, from closing speed *v*, lasts
*T = 2d / v*. This holds for a constant deceleration and for a half-sine pulse, assuming no rebound; for the
half-sine, displacement during the pulse is *vT/2*. The half-sine peak is *A = πv² / (4d)*.

**Table A.2 — Pulse duration and half-sine peak from crush distance and speed (calculation, not measurement)**

| closing speed | d = 1 cm | d = 3 cm | d = 6 cm | d = 11 cm |
|---|---|---|---|---|
| 2.0 m/s (7.2 km/h) | 10.0 ms, 32 g | 30.0 ms, 11 g | 60.0 ms, 5.3 g | 110 ms, 2.9 g |
| 3.0 m/s (10.8 km/h) | 6.7 ms, 72 g | 20.0 ms, 24 g | 40.0 ms, 12 g | 73 ms, 6.6 g |
| 5.56 m/s (20 km/h) | 3.6 ms, 248 g | 10.8 ms, 83 g | 21.6 ms, 41 g | 39.6 ms, 22 g |

A full-size car crushing about 0.5 m at 56 km/h gives *T* ≈ 64 ms, consistent with the ~70 ms median pulse in
VZCrash. An RC car whose bumper and chassis yield about 1 cm gives 3.6–10 ms across its speed range. That is
under the 40 ms floor by a factor of 4–11 before any other test is applied. A 40 ms pulse needs a crush stroke
of *d = vT/2*: 4 cm at 2 m/s, 6 cm at 3 m/s, and 11 cm at full throttle (5.56 m/s).

### A.2.2 Sampling and the serve-path filter

Duration is not measured on the physical pulse. It is measured on what the system records.

- **Sampling.** The firmware samples at 100 Hz, so a pulse shorter than 10 ms spans at most one sample. The
  MPU6050's digital low-pass filter is left at its reset value (DLPF_CFG = 0, about 260 Hz accelerometer
  bandwidth), so each reading is near-instantaneous. A few-millisecond pulse is caught near its peak only when a
  read happens to land inside it, and is otherwise missed. At this scale the recorded peak is set by sampling
  phase as much as by physics.
- **Filter.** The server applies a 4th-order 20 Hz zero-phase Butterworth low-pass (`app._lp`) before the gate.
  For a rectangular +3 g pulse on the gravity axis, the filtered peak keeps 40% of the excess at 10 ms, 70% at
  20 ms and 100% from 30 ms upward. Reproduce with `app._lp` on `np.ones(500)` plus a pulse.
- **Sensor range.** Table A.2's peaks for an unmodified RC car (tens to hundreds of g) exceed the ±16 g per-axis
  range, so the recorded peak is also clipped.

**Consequence.** Every metric in this appendix is computed on the serve-path signal the gate actually sees,
never on idealised physics. Table A.2 motivates the problem but does not set any threshold.

### A.2.3 Impulse and delta-v

The production Severe cut, 0.959 g·s, is Δv = 9.40 m/s. An RC car at 20 km/h carries 5.56 m/s. With coefficient
of restitution *e*, Δv = (1 + *e*)·*v*:
- **Zero rebound:** Δv = 5.56 m/s, short of the cut by a factor of 1.69.
- **Rebound:** the cut is reached only if *e* > 0.69. Rigid-barrier RC impacts can rebound, so this
  cannot be ruled out without measurement.

However, the gate's impulse integrates only samples above the run level, after the 20 Hz filter (§A.2.2). For a
pulse of a few milliseconds it captures a small, sampling-dependent fraction of the true Δv. The analysis reports
the measured ratio of impulse-derived Δv to logged approach speed, so this is quantified rather than assumed.

### A.2.4 The model vote

A crash classification needs **both** the model (P(crash) ≥ 0.42) and the gate. The model was trained on
full-size vehicle windows and must not be retrained or altered. Whether it assigns crash probability to RC
impacts is an empirical question. The analysis reports the model's vote rate per class, and a separate
confusion matrix for the gate alone (the firmware's local path) and for the full server decision. If the model
vetoes RC impacts, `demo_rc` cannot produce crash classifications on the server whatever its thresholds, and
this appendix will say so.

## A.3 The profile mechanism

### A.3.1 One definition, two consumers

`artifacts/threshold_profiles.json` is the only place thresholds are written. It holds `production` (Table A.1,
pinned by `tests/test_profiles.py::test_production_thresholds_are_the_thesis_values`) and `demo_rc`.
- **Server:** `profiles.py` loads the file, and `app.py` applies the active profile in `crash_signature()`.
- **Firmware:** `tools/gen_firmware_profiles.py` compiles the same file, plus the server's exact filter
  coefficients, into `firmware/ESP32_Complete_3LED_System/threshold_profiles.h`. The test suite fails if the
  committed header differs from the JSON.
- **Parity:** the firmware gate (`signature_gate.h`: odd-padded `sosfiltfilt`, excursion, impulse) is compiled on
  the host and compared window by window with the Python gate. Across 175 windows (25 API fixtures and 150
  randomised pulses) under two threshold sets, peak and impulse agree within the asserted 10⁻⁴ g (largest
  difference seen during development: 1.6 × 10⁻⁶ g), excursion exactly, and signature and class identically.

### A.3.2 Selection and fallback

| Where | How selected | Default | Unset / unrecognised / not derived |
|---|---|---|---|
| Server | environment variable `THRESHOLD_PROFILE` | `production` | resolves to `production`; `profile_note` states why |
| Firmware | compile-time `#define THRESHOLD_PROFILE_DEMO_RC` | `production` | not derived: compiler `#warning`, builds `production`, banner prints the note |

A missing, misspelt or incomplete profile can never resolve to `demo_rc`. The production duration floor keeps
its pre-existing `SIG_TRANSIENT_MIN_MS` override, which has no effect on `demo_rc`.

### A.3.3 Visibility

- **`/predict` and `/health`:** every response carries `profile`, `profile_thresholds`, `profile_requested` and
  `profile_note`. These are response additions only; the request contract is unchanged, and the existing
  138-check API suite passes unmodified.
- **`/api/v1/events`:** every response carries the same fields plus an `X-Threshold-Profile` header.
- **Incident rows:** each row stores `server_profile`, the server's thresholds at ingest, and `device_profile`.
- **Firmware:** the banner prints the profile in full with all numeric thresholds and the Δv equivalent, plus a
  "SCALED DEMONSTRATION PROFILE" warning when it is not `production`. Every classification line reads
  `[CLASSIFY] source=… profile=…`, and a device/server profile mismatch prints `[PROFILE] MISMATCH`.
- **Dashboard: not implemented in this change.** The fields it needs are available from `GET /api/v1/events`.

### A.3.4 The one non-threshold difference: Severe grading

`production` grades Severe on the server by P(Severe | crash) ≥ 0.66 (Phase 5). `demo_rc` sets
`severity_grading: "impulse"`, grading Severe when the gate's impulse meets the profile's cut. The reason is that
the model's Severe probability was learned from labels defined at the VZCrash 85th-percentile impulse
(Δv ≈ 9.4 m/s), which §A.2.3 shows an RC car cannot produce. The model itself, its calibration and the crash
threshold 0.42 are identical under both profiles. The firmware's local fallback grades by impulse under both.

## A.4 Collection protocol

### A.4.1 Before collecting

1. **Persistence.** Before this change, no raw window was stored anywhere, which is why earlier incidents could
   not be re-analysed. Every incident now stores its full 500-sample window, both classifications and both
   profiles (`POST /api/v1/events`). A failed upload prints the complete event as an `[UNSENT_EVENT]` serial line,
   which `tools/ingest_serial_log.py` re-posts.
2. **Durable storage.** The default store is `sqlite:///data/events.db` inside the container, and Railway wipes
   it on redeploy. Mount a Railway volume at `/app/data`, or set `EVENTS_DATABASE_URL`, before collecting. Set
   `EVENTS_API_KEY` on the server and in `secrets.h` so the store cannot be written by third parties.
3. **Firmware.** Flash the default (production) build. Collection does not depend on the profile, because every
   window is re-measured on the serve path during analysis. Capture the full serial log of every session to
   `data/rc_collection/session_logs/`.
4. **Mount.** Fix the ESP32/MPU6050 rigidly to the chassis, Z axis up; the boot check reports the orientation.
   Record the mounting, vehicle mass, battery and tyre condition once per session.

### A.4.2 Classes and run conditions

Collect **at least 30 events per class**. The analysis refuses to write a profile from fewer.

| Class (`label`) | Manoeuvres | Capture |
|---|---|---|
| `normal` | straight driving, cornering, hard acceleration, braking, running over obstacles, picking the car up and putting it down | `capture` during the manoeuvre (window centred on the command); any trigger during the manoeuvre is stored automatically |
| `impact_low` | low throttle into a padded barrier | automatic trigger; if none fires within 2 s of impact, run `capture` immediately and note it |
| `impact_high` | full throttle into a rigid barrier | as above |

Before each run, set:

```
run speed=<km/h> barrier=<none|padded|rigid> angle=<deg, 0 = head-on> crush=<none|foam|lattice|spring> cond=<id> notes=<free text>
label <normal|impact_low|impact_high>
```

`collect on` refuses to start without speed, barrier, angle and crush. The label clears after every stored
event, so each run must be labelled deliberately. Measure speed rather than estimating it: time the car over a
marked 1 m approach from video at ≥ 60 fps, or use a GPS/wheel-speed log if fitted. Record how it was measured in
`notes`. Vary impact angle (0°, ±30°) within each impact class, and interleave classes rather than recording
them in blocks, so battery depletion does not confound class.

### A.4.3 Crush-structure runs

For each condition id (`cond`, a fixed speed + barrier + angle), record at least 10 impacts with `crush=none`
and at least 10 with the structure fitted (`foam`, `lattice` or `spring`), alternating in blocks of five. Record
the structure's material, thickness and available stroke in `notes`. Table A.2 gives the design target: a 40 ms
pulse needs about 4 cm of stroke at 2 m/s, 6 cm at 3 m/s and 11 cm at full throttle. Report the result whatever
it shows (§A.9).

### A.4.4 After each session

```
python tools/export_events.py --event-type rc_collection      # -> data/rc_collection/events.jsonl
python analysis/rc_demo_profile.py --vzcrash ../phase2/results/vzcrash_windows.csv
```

## A.5 Derivation rules (fixed before data collection)

The rules are coded in `analysis/rc_demo_profile.py`, applied to serve-path signals, and were fixed before any
RC data existed. Each rule mirrors the logic behind the corresponding production value.

| Threshold | Production logic (VZCrash) | `demo_rc` rule (RC data) |
|---|---|---|
| Peak band floor | lower edge of the crash-peak mass; 0% of crash events below 2 g | the value maximising Youden's J = (impacts ≥ t) − (handling events ≥ t) on a 0.05 g grid; ties resolve to the midpoint of the tied interval (the boundary halfway between the populations) |
| Peak band ceiling | excludes the ≥ 7 g manoeuvre/drop population while keeping 99.93% of crashes | given the floor, the ceiling maximising J over [floor, t), same tie rule, capped at the ±16 g sensor range; the fraction of impacts inside is reported as VZCrash's 99.93% was |
| Duration floor | 17.2% of crash events below 40 ms | largest 10 ms multiple with ≤ 17.2% of measured impacts below it, measured as the longest run above the derived peak floor |
| Duration ceiling | 1.8% of crash events above 250 ms | smallest 10 ms multiple with ≤ 1.8% of impacts above it |
| Severe impulse | 85th percentile of crash impulse (Δv 9.40 m/s) | the impulse maximising J between `impact_high` and `impact_low` (max-margin midpoint), stated as Δv = impulse × 9.80665 m/s. The 85th percentile of RC impact impulse is reported alongside for comparison with the production rule |

**Evaluation.** Reported on the collected set:
- a 3 × 3 confusion matrix (true `normal` / `impact_low` / `impact_high` against predicted Normal / Moderate /
  Severe), crash recall, false-positive rate on `normal`, and the per-class rates;
- two predictions: gate only (firmware local path, including whether the device trigger would have fired) and
  full server (trigger AND model vote AND gate);
- two estimates: in-sample, which is optimistic because the thresholds were fit to the same events, and 5-fold
  stratified cross-validation with the thresholds re-derived on each training fold. **The cross-validated
  figures are the ones to quote.**

The production profile is also applied to the same events, to show its unreachability empirically.

## A.6 Measured distributions

**Not yet measured.** When run, the analysis writes the table below to `analysis/out/demo_rc_report.md`: median,
IQR, 5th and 95th percentile per class, for raw and serve-path peak, run length ≥ 2 g, run length ≥ derived floor,
impulse, model P(crash) and logged speed. It also writes the figures corresponding to Phase 2's `v1_crash_peak_hist`,
`v2_crash_vs_maneuver_duration` and `v3_peak_vs_duration`, placed side by side with VZCrash, plus an impulse
histogram for low- vs high-energy impacts:
- `v1_peak_hist_vzcrash_vs_rc.png`
- `v2_duration_vzcrash_vs_rc.png`
- `v3_peak_vs_duration_vzcrash_vs_rc.png`
- `v4_impulse_low_vs_high.png`

| class | metric | n | median | IQR | p5 | p95 |
|---|---|---|---|---|---|---|
| normal | … | — | — | — | — | — |
| impact_low | … | — | — | — | — | — |
| impact_high | … | — | — | — | — | — |

## A.7 Derived thresholds

**Not yet derived.** `analysis/rc_demo_profile.py --write-profile` writes the values below into
`artifacts/threshold_profiles.json` with their provenance: dataset SHA-256, events per class, rule statements,
coverage and cross-validated recall/FPR. It then regenerates the firmware header. It refuses when any class has
fewer than 30 events or when any row is synthetic.

| Threshold | Value | Derivation result |
|---|---|---|
| peak band | — | fraction of impacts inside: — ; handling events inside: — |
| duration window | — | impacts below floor: — ; above ceiling: — |
| severe impulse | — | Δv: — m/s; J: — ; 85th-percentile comparison: — |

## A.8 Confusion matrix on the collected set

**Not yet measured.**

| true \ predicted | Normal | Moderate | Severe |
|---|---|---|---|
| normal | — | — | — |
| impact_low | — | — | — |
| impact_high | — | — | — |

Crash recall: —. False-positive rate: —. Cross-validated, gate only and server: —. Production profile on the same
events: —.

## A.9 Crush-structure measurement

**Not yet measured.** For each condition id, the analysis reports:
- median raw and serve-path peak with and without the structure;
- median pulse duration: raw ≥ 2 g, serve-path ≥ 2 g (production run level) and serve-path ≥ derived floor;
- Mann–Whitney *p*;
- the fraction of impacts reaching production's 40 ms at 2 g on the serve path.

If the structure brings the pulse near 40 ms, the duration rule in §A.5 will produce a floor close to production's
from the data themselves, a stronger position than wholesale rescaling. If it does not, that result is reported
here unchanged.

| cond | structure | n (without/with) | peak without → with | duration ≥ 2 g without → with | reaching 40 ms without → with |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

## A.10 Exclusion from performance statistics

`events.performance_exclusion_reason()` is the single eligibility rule for `/api/v1/stats`. A row enters a recall,
precision, F1 or confusion figure only if **all** of the following hold:
- `event_type == "trigger"`, which excludes `manual_panic` and `rc_collection`;
- `server_profile == "production"`;
- `device_profile` is `production` or unreported;
- the row has a verified ground-truth label.

Every other row is counted under `excluded`, with its reason.

`tests/test_profiles.py::test_demo_profile_incidents_excluded_from_performance_statistics` stores four eligible
production incidents and records their statistics. It then adds six rows, each labelled so that including it
would change the figures:
- a `manual_panic`;
- a device built with `demo_rc`;
- an `rc_collection` event;
- an incident classified while the server ran `demo_rc`;
- a production device reporting to a `demo_rc` server;
- an unlabelled trigger.

It asserts that the confusion matrix, per-class metrics, macro-F1 and crash-detection metrics are unchanged. Output
of `python -m pytest tests/test_profiles.py -v -s -k "excluded_from_performance or exclusion_rule"` (2026-09-16):

```
tests/test_profiles.py::test_demo_profile_incidents_excluded_from_performance_statistics
performance statistics after adding excluded rows:
{
  "confusion_matrix": {"matrix": [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
                       "rows_true_cols_pred": ["Normal", "Moderate", "Severe"]},
  "crash_detection": {"f1": 0.8, "precision": 1.0, "recall": 0.6667},
  "eligibility_rule": "event_type == 'trigger' AND server_profile == 'production' AND device_profile in ('production', null) AND ground_truth verified",
  "excluded": {"manual_panic": 1, "no_verified_label": 1, "non_production_profile": 3, "rc_collection": 1},
  "included_n": 4,
  "macro_f1": 0.7778,
  ...
}
PASSED
tests/test_profiles.py::test_exclusion_rule_table PASSED
====================== 2 passed, 28 deselected in 4.32s =======================
```

These four incidents are test fixtures used to exercise the rule; they are not results. The full suite
(`python -m pytest tests/test_profiles.py -v`, output in `docs/test_output_profiles.txt`) also covers:
- fallback resolution for unset, unknown and underived profiles;
- identity of the production gate with the pre-profile implementation, and unchanged predictions on every API
  fixture;
- profile fields on `/predict`, `/health` and `/api/v1/events`;
- raw-window storage;
- firmware header freshness and C++/Python gate parity;
- the analysis script's refusal to write a profile from synthetic or underpowered data.

## A.11 Reproduction

```
python -m pytest tests/test_profiles.py -v                     # mechanism, exclusion, parity (needs g++ for parity)
python test_api.py --url http://localhost:5000                 # unchanged /predict contract (138 checks)
python tools/gen_firmware_profiles.py --check                  # firmware header matches the JSON
python tools/export_events.py --event-type rc_collection
python analysis/rc_demo_profile.py --vzcrash ../phase2/results/vzcrash_windows.csv [--write-profile]
```

## A.12 Limitations

- **Pending measurement.** Every quantitative claim about RC impacts in §A.2 is a calculation; §A.6–§A.9 contain
  no measurements yet.
- **Sampling limits.** At 100 Hz with the MPU6050 DLPF at its reset value, pulses under about 10 ms are
  under-sampled and their recorded peak depends on sampling phase. Changing the DLPF or sample rate would alter
  the production data path and is out of scope without re-validating the production results.
- **Model veto.** If the model does not recognise RC impacts (§A.2.4), the server cannot demonstrate crash
  classification under any `demo_rc` thresholds. Only the firmware's model-free local path would.
- **Sample size.** With about 30 events per class, a ceiling or floor set at a 1.8% or 17.2% position is
  determined by one or a few events. The cross-validated estimates and the per-fold thresholds
  (`fold_thresholds` in `demo_rc_derivation.json`) show how stable each threshold is.
