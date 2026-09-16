# RC-car collection dataset

**Status (2026-09-16): no events collected yet.** This directory holds the measured dataset from which the `demo_rc`
threshold profile is derived. It stays empty until the protocol in `DEMO_PROFILE.md` §A.4 has been run on the
physical demonstrator. Nothing here may be synthetic. Test fixtures live in `tests/rc_synthetic.py` and carry
`"synthetic": true`, and `analysis/rc_demo_profile.py --write-profile` refuses to use them.

## Files

| file | produced by | content |
|---|---|---|
| `events.jsonl` | `python tools/export_events.py --event-type rc_collection` | one incident per line, as stored by `/api/v1/events` |
| `session_logs/*.log` | serial monitor capture during each session | the raw serial log, kept as the lab notebook; `[UNSENT_EVENT]` lines are recoverable with `tools/ingest_serial_log.py` |

Commit `events.jsonl` together with the session logs, `analysis/out/demo_rc_derivation.json` and the regenerated
`artifacts/threshold_profiles.json`, so the dataset hash recorded in the profile provenance points at a committed file.

## Row schema (fields used by the analysis)

| field | type | meaning |
|---|---|---|
| `id` | int | incident id in the store |
| `event_type` | `"rc_collection"` | other types are ignored |
| `ground_truth` | `normal` \| `impact_low` \| `impact_high` | class label, set on the device before the run (`label ...`) or later with `PATCH /api/v1/events/<id>`; unlabelled rows are skipped |
| `run_conditions` | object | `speed_kmh`, `barrier` (`none`/`padded`/`rigid`), `angle_deg`, `crush` (`none`/`foam`/`lattice`/`spring`), `cond` (condition id pairing with/without-crush runs), `notes` |
| `raw_window` | object | `ax ay az` (g) and `gx gy gz` (deg/s), 500 samples each at 100 Hz, oldest first |
| `window_sha256` | hex | SHA-256 of the canonical JSON of `raw_window` |
| `device_profile` / `server_profile` | string | threshold profile compiled into the firmware / active on the server |
| `device_metrics` | object | `capture` (`trigger`/`manual`), `trigger_mag_g`, `raw_peak_g`, `raw_peak_index` |
| `server_severity_class`, `peak_g`, `excursion_ms`, `impulse_gs`, `p_crash` | | server classification at ingest (under `server_profile`) |
