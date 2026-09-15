"""
Build the API test fixtures and the Postman collection from ONE source of truth.

  tests/fixtures.json      - cases + expected responses (consumed by test_api.py)
  Postman_v2_Tests.json    - the same cases for the Postman Collection Runner

Cases:
  * designed synthetic windows (in g, gravity on +Z, 100 Hz) with an INTENDED
    class. Generation aborts if the served pipeline disagrees with the intent,
    so a model/threshold change that breaks a designed case cannot ship silently.
  * real windows drawn at random (seed 2026, never filtered by prediction) from
    the group-disjoint TEST hold-out of phase5 (VZCrash + Phase-1). Their expected
    response is what this validated build returns; their TRUE label is recorded
    and reported separately, so misclassifications stay visible.
  * contract cases (missing field, wrong length, non-JSON, NaN, Inf, null).

Usage:  python tests/make_fixtures.py [path/to/phase5/data/fixture_windows.json]
"""
from __future__ import annotations
import json, sys, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app  # noqa: E402

PROD = "https://accident-severity-api-production.up.railway.app"
FIX_SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\okrah\OneDrive\Desktop\phase5\data\fixture_windows.json")
CH = ["ax", "ay", "az", "gx", "gy", "gz"]
T = np.arange(500)


def rest(rng, noise=0.05):
    return dict(ax=rng.normal(0, noise, 500), ay=rng.normal(0, noise, 500), az=1.0 + rng.normal(0, noise, 500),
                gx=rng.normal(0, 2, 500), gy=rng.normal(0, 2, 500), gz=rng.normal(0, 2, 500))


def pulse(center, fwhm_ms, amp):
    sigma = (fwhm_ms / 10.0) / 2.355
    return amp * np.exp(-0.5 * ((T - center) / sigma) ** 2)


def synth(kind, seed):
    rng = np.random.default_rng(seed)
    w = rest(rng, 0.05 if kind in ("normal", "spike") else 0.1)
    if kind == "spike":                                   # dropped device: 12 g for 50 ms
        w["ax"][250:255] += 12.0
    elif kind == "sustained":                             # 9 g held for 2.5 s: manoeuvre/artifact
        w["ax"][150:400] += 9.0
    elif kind == "moderate":                              # single 4 g / 90 ms impact
        p = pulse(250, 90, 3.6); w["ax"] += p; w["ay"] += 0.4 * p; w["gz"] += 60 * p
    elif kind == "severe":                                # two impacts 1.2 s apart: high impulse, each pulse <= 250 ms
        p = pulse(190, 150, 5.5) + pulse(310, 150, 5.0)
        w["ax"] += p; w["ay"] += 0.5 * p; w["gz"] += 80 * p
    elif kind == "moderate_ms2":                          # same impact, client sends m/s² -> unit guard must rescale
        w = synth("moderate", seed)
        for k in ("ax", "ay", "az"):
            w[k] = w[k] * 9.80665
        return w
    elif kind == "short":                                 # 3 g / 30 ms: below the 40 ms floor
        p = pulse(250, 30, 2.6); w["ax"] += p; w["ay"] += 0.4 * p
    return {k: v for k, v in w.items()}


SYNTH = [
    ("Synthetic: normal driving", "normal", 11, "Normal"),
    ("Synthetic: dropped device, 12 g / 50 ms spike", "spike", 12, "Normal"),
    ("Synthetic: sustained 9 g for 2.5 s", "sustained", 13, "Normal"),
    ("Synthetic: single impact ~4 g / 90 ms", "moderate", 14, "Moderate"),
    ("Synthetic: two impacts 1.2 s apart, 5.5 g + 5 g, 150 ms each", "severe", 15, "Severe"),
    ("Synthetic: 3 g / 30 ms pulse (below 40 ms floor)", "short", 16, "Normal"),
    ("Synthetic: single impact ~4 g / 90 ms sent in m/s² (unit guard)", "moderate_ms2", 14, "Moderate"),
]


def payload(w):
    return {k: [round(float(v), 6) for v in np.asarray(w[k])] for k in CH}


def expected_for(p):
    r = app.run_inference(*(np.asarray(p[k], float) for k in CH))
    return {k: r[k] for k in ("severity_class", "severity_name", "p_crash", "label_source", "accident_confirmed")} | \
           {"signature_match": r["crash_signature"]["signature_match"], "excursion_ms": r["crash_signature"]["excursion_ms"],
            "impulse_gs": r["crash_signature"]["impulse_gs"], "peak_g": r["crash_signature"]["peak_g"]}


def main():
    cases = []
    for name, kind, seed, intent in SYNTH:
        p = payload(synth(kind, seed)); exp = expected_for(p)
        if exp["severity_name"] != intent:
            raise SystemExit(f"designed case '{name}' intended {intent} but the served pipeline returns "
                             f"{exp['severity_name']} ({exp}) - fix the model/thresholds or the design")
        cases.append(dict(name=name, kind="synthetic", true_label=intent, payload=p, expected=exp))

    real = json.load(open(FIX_SRC))
    for r in real:
        p = r["window"]; exp = expected_for(p)
        cases.append(dict(name=f"Real hold-out: {r['category']} ({r['recording_id']} w{r['window_idx']})", kind="real",
                          true_label=app.CLASS_NAMES[r["true_label"]], payload=p, expected=exp))

    base = payload(synth("normal", 11))
    contract = [
        dict(name="Contract: missing gz -> 400", body={k: v for k, v in base.items() if k != "gz"}, status=400, error_includes="gz"),
        dict(name="Contract: 100 samples -> 400", body={k: v[:100] for k, v in base.items()}, status=400, error_includes="500"),
        dict(name="Contract: null sample -> 400", body={**base, "ay": [None] + base["ay"][1:]}, status=400, error_includes="NaN or Inf"),
        dict(name="Contract: NaN sample -> 400", raw=json.dumps({**base, "ax": [float("nan")] + base["ax"][1:]}), status=400, error_includes="NaN or Inf"),
        dict(name="Contract: Infinity sample -> 400", raw=json.dumps({**base, "az": base["az"][:-1] + [float("inf")]}), status=400, error_includes="NaN or Inf"),
        dict(name="Contract: text/plain body -> 400", raw="not json", content_type="text/plain", status=400, error_includes="application/json"),
    ]

    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "unknown"
    fixtures = dict(generated_from=dict(model=app.MODEL_NAME, crash_threshold=app.CRASH_ALERT_THRESHOLD,
                                        severe_ratio_threshold=app.SEVERE_RATIO_THRESHOLD, git_head_at_generation=sha),
                    health=dict(model=app.MODEL_NAME, n_features=len(app.FEATURES), taxonomy="signature+impulse (v2)",
                                crash_alert_threshold=app.CRASH_ALERT_THRESHOLD, severe_ratio_threshold=app.SEVERE_RATIO_THRESHOLD),
                    cases=cases, contract=contract)
    (ROOT / "tests").mkdir(exist_ok=True)
    json.dump(fixtures, open(ROOT / "tests" / "fixtures.json", "w"))
    write_postman(fixtures)
    n_real = sum(c["kind"] == "real" for c in cases)
    agree = sum(c["kind"] == "real" and c["expected"]["severity_name"] == c["true_label"] for c in cases)
    print(f"fixtures: {len(cases)} classification cases ({n_real} real, true-label agreement {agree}/{n_real}), {len(contract)} contract cases")
    for c in cases:
        print(f"  {c['expected']['severity_name']:8s} (true {c['true_label']:8s}) p_crash={c['expected']['p_crash']:.3f} "
              f"sig={c['expected']['signature_match']!s:5s} {c['name']}")


def write_postman(fx):
    h = fx["health"]
    items = [{
        "name": "00. Health",
        "request": {"method": "GET", "header": [], "url": {"raw": "{{base_url}}/health", "host": ["{{base_url}}"], "path": ["health"]}},
        "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": [
            'pm.test("200 OK", () => pm.response.to.have.status(200));',
            'const d = pm.response.json();',
            'pm.test("healthy", () => pm.expect(d.status).to.eql("healthy"));',
            f'pm.test("model {h["model"]}", () => pm.expect(d.model).to.eql("{h["model"]}"));',
            f'pm.test("taxonomy v2", () => pm.expect(d.taxonomy).to.eql("{h["taxonomy"]}"));',
            f'pm.test("25 features", () => pm.expect(d.n_features).to.eql({h["n_features"]}));',
            f'pm.test("crash threshold", () => pm.expect(d.crash_alert_threshold).to.be.closeTo({h["crash_alert_threshold"]}, 1e-6));',
            f'pm.test("severe threshold", () => pm.expect(d.severe_ratio_threshold).to.be.closeTo({h["severe_ratio_threshold"]}, 1e-6));',
        ]}}]}]
    for i, c in enumerate(fx["cases"], start=1):
        e = c["expected"]
        tests = [
            'pm.test("200 OK", () => pm.response.to.have.status(200));',
            'const d = pm.response.json();',
            f'pm.test("severity {e["severity_name"]}", () => pm.expect(d.severity_name).to.eql("{e["severity_name"]}"));',
            f'pm.test("label_source {e["label_source"]}", () => pm.expect(d.label_source).to.eql("{e["label_source"]}"));',
            f'pm.test("p_crash matches validated build", () => pm.expect(d.p_crash).to.be.closeTo({e["p_crash"]}, 0.002));',
            f'pm.test("signature_match {str(e["signature_match"]).lower()}", () => pm.expect(d.crash_signature.signature_match).to.eql({str(e["signature_match"]).lower()}));',
            f'console.log("true label: {c["true_label"]} | returned: " + d.severity_name);',
        ]
        items.append({"name": f"{i:02d}. {c['name']}",
                      "request": {"method": "POST", "header": [{"key": "Content-Type", "value": "application/json"}],
                                  "body": {"mode": "raw", "raw": json.dumps(c["payload"]), "options": {"raw": {"language": "json"}}},
                                  "url": {"raw": "{{base_url}}/predict", "host": ["{{base_url}}"], "path": ["predict"]},
                                  "description": f"True label: {c['true_label']}. Expected response from the validated build: {e['severity_name']}."},
                      "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": tests}}]})
    for j, c in enumerate(fx["contract"], start=len(items)):
        raw = c.get("raw", json.dumps(c.get("body")))
        items.append({"name": f"{j:02d}. {c['name']}",
                      "request": {"method": "POST", "header": [{"key": "Content-Type", "value": c.get("content_type", "application/json")}],
                                  "body": {"mode": "raw", "raw": raw},
                                  "url": {"raw": "{{base_url}}/predict", "host": ["{{base_url}}"], "path": ["predict"]}},
                      "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": [
                          f'pm.test("{c["status"]}", () => pm.response.to.have.status({c["status"]}));',
                          f'pm.test("error mentions {c["error_includes"]}", () => pm.expect(pm.response.json().error).to.include("{c["error_includes"]}"));']}}]})
    coll = {"info": {"name": "Accident Severity API - v2 (Phase 5 operating point)",
                     "description": "Generated by tests/make_fixtures.py from tests/fixtures.json. Designed Normal/Moderate/Severe cases, "
                                    "random real hold-out windows (true label logged per request), and contract cases. "
                                    "All payloads are in g with gravity on +Z at 100 Hz.",
                     "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
            "variable": [{"key": "base_url", "value": PROD, "type": "string"}], "item": items}
    json.dump(coll, open(ROOT / "Postman_v2_Tests.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
