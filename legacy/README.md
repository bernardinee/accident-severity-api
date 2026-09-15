# Legacy test assets (do not use against the current API)

These files test the **retired v1 taxonomy** (peak ≥ 7 g → Severe) and are kept only for history.

| File | Why it is invalid now |
|---|---|
| `Postman_Tests_FINAL.json`, `Postman_Tests_DEBUG.json`, `generate_postman_scenarios_DEBUG.py` | Mixed units: gravity on `az` is 9.8 (m/s²) while crash pulses are added in g. The API's unit guard divides the whole window by 9.81, so a "22 g" pulse arrives as ~2.2 g and **every scenario returns Normal**. Expectations also encode the retired peak-height rule. |
| `Postman_Tests_Fixed.json`, `Postman_Accident_Detection_Tests.json` | Send only `ax/ay/az`; the API requires all six channels → HTTP 400 on every request (also under v1). |
| `generate_postman_v2_phase3.py` | Phase 3 v2 suite; superseded by `tests/make_fixtures.py`, which adds a Severe case, real hold-out windows and NaN/Inf/null contract checks. |
| `api_test_results_prod_v1_2026-05-14.txt` | Result of the old test suite against the v1 model (`xgboost_no_proxy`). |

Current suite: `tests/make_fixtures.py` → `tests/fixtures.json` → `test_api.py`, `smoke_test.py`, `Postman_v2_Tests.json`.
See `DIAGNOSIS.md` for the investigation.
