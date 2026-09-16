"""Threshold profiles, incident store, performance-statistic exclusion, firmware parity.

    python -m pytest tests/test_profiles.py -v
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import app
import events
import profiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gen_firmware_profiles as gen  # noqa: E402

FIXTURES = json.loads((ROOT / "tests" / "fixtures.json").read_text())
CH = ("ax", "ay", "az", "gx", "gy", "gz")
DEMO_TEST = profiles.Profile("demo_rc", 1.2, 12.0, 10.0, 150.0, 0.05, "impulse", "demo_rc", None)


def derived_profiles_file(tmp_path, **demo):
    defs = profiles.load_definitions()
    values = dict(status="derived", peak_min_g=1.2, peak_max_g=12.0, dur_min_ms=10.0, dur_max_ms=150.0,
                  severe_impulse_gs=0.05)
    defs["profiles"]["demo_rc"].update({**values, **demo})
    p = tmp_path / "threshold_profiles.json"
    p.write_text(json.dumps(defs))
    return p


# ── production is unchanged and is the default ────────────────────────────────
def test_production_thresholds_are_the_thesis_values():
    prod = profiles.load_definitions()["profiles"]["production"]
    assert (prod["peak_min_g"], prod["peak_max_g"], prod["dur_min_ms"], prod["dur_max_ms"],
            prod["severe_impulse_gs"]) == (2.0, 7.0, 40.0, 250.0, 0.959)
    assert prod["severity_grading"] == "model_ratio"
    assert profiles.load_definitions()["default"] == "production"


@pytest.mark.parametrize("requested,note_has", [
    (None, None), ("", None), ("production", None), ("  PRODUCTION ", None),
    ("demo", "unrecognised"), ("prod", "unrecognised"), ("DEMO-RC", "unrecognised"),
    ("demo_rc", "not derived"),
])
def test_unset_unknown_or_underived_profiles_resolve_to_production(requested, note_has):
    p = profiles.resolve(requested=requested, env={})
    assert p.name == "production"
    assert (p.note is None) if note_has is None else (note_has in p.note)


def test_env_selects_profile_and_derived_demo_is_honoured(tmp_path):
    f = derived_profiles_file(tmp_path)
    assert profiles.resolve(path=f, env={"THRESHOLD_PROFILE": "demo_rc"}).name == "demo_rc"
    assert profiles.resolve(path=f, env={}).name == "production"
    assert profiles.resolve(path=f, env={"THRESHOLD_PROFILE": "bogus"}).name == "production"


def test_partially_derived_demo_falls_back_to_production(tmp_path):
    f = derived_profiles_file(tmp_path, severe_impulse_gs=None)
    p = profiles.resolve(path=f, env={"THRESHOLD_PROFILE": "demo_rc"})
    assert p.name == "production" and "not derived" in p.note


def test_duration_override_applies_to_production_only(tmp_path):
    f = derived_profiles_file(tmp_path)
    assert profiles.resolve(path=f, env={"SIG_TRANSIENT_MIN_MS": "20"}).dur_min_ms == 20.0
    assert profiles.resolve(path=f, env={"SIG_TRANSIENT_MIN_MS": "20", "THRESHOLD_PROFILE": "demo_rc"}).dur_min_ms == 10.0


def test_app_default_profile_is_production():
    assert app.ACTIVE_PROFILE.name == "production" and app.ACTIVE_PROFILE.note is None
    assert (app.PEAK_MIN_G, app.PEAK_MAX_G, app.TRANSIENT_MIN_MS, app.TRANSIENT_MAX_MS) == (2.0, 7.0, 40.0, 250.0)


def _pre_profile_crash_signature(ax, ay, az):
    """app.crash_signature exactly as it was before profiles (git 780412e)."""
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    peak = float(mag.max())
    above = mag >= 2.0
    longest = run = 0
    for a in above:
        run = run + 1 if a else 0
        longest = max(longest, run)
    longest_ms = longest * 1000.0 / 100
    impulse = float(np.sum(mag[above] - 1.0) / 100) if above.any() else 0.0
    is_sig = (2.0 <= peak < 7.0 and 40.0 <= longest_ms <= 250.0)
    return peak, longest_ms, impulse, bool(is_sig)


def test_production_gate_identical_to_pre_profile_gate():
    rng = np.random.default_rng(0)
    wins = [[np.asarray(c["payload"][k], float) for k in ("ax", "ay", "az")] for c in FIXTURES["cases"]]
    for _ in range(200):
        w = [rng.normal(0, .05, 500), rng.normal(0, .05, 500), 1 + rng.normal(0, .05, 500)]
        s, n = rng.integers(0, 480), rng.integers(1, 40)
        w[rng.integers(0, 3)][s:s + n] += rng.uniform(0.5, 10)
        wins.append(w)
    for w in wins:
        f = [app._lp(x) for x in w]
        assert app.crash_signature(*f) == _pre_profile_crash_signature(*f)


def test_production_predictions_unchanged_on_fixtures():
    for c in FIXTURES["cases"]:
        r = app.run_inference(*(np.asarray(c["payload"][k], float) for k in CH))
        e = c["expected"]
        assert (r["severity_name"], r["label_source"], r["crash_signature"]["signature_match"]) == \
               (e["severity_name"], e["label_source"], e["signature_match"]), c["name"]
        assert abs(r["p_crash"] - e["p_crash"]) <= 0.002


# ── visibility ────────────────────────────────────────────────────────────────
def test_predict_and_health_report_profile():
    client = app.app.test_client()
    body = FIXTURES["cases"][0]["payload"]
    r = client.post("/predict", json=body).get_json()
    assert r["profile"] == "production" and r["profile_requested"] is None and r["profile_note"] is None
    assert r["profile_thresholds"]["peak_max_g"] == 7.0 and r["profile_thresholds"]["dur_min_ms"] == 40.0
    assert client.get("/health").get_json()["profile"] == "production"


def test_predict_request_contract_unchanged():
    client = app.app.test_client()
    for c in FIXTURES["contract"]:
        if "raw" in c:
            resp = client.post("/predict", data=c["raw"], content_type=c.get("content_type", "application/json"))
        else:
            resp = client.post("/predict", data=json.dumps(c.get("body")), content_type=c.get("content_type", "application/json"))
        assert resp.status_code == c["status"] and c["error_includes"] in resp.get_json().get("error", ""), c["name"]


def test_predict_under_demo_profile_says_so(monkeypatch):
    monkeypatch.setattr(app, "ACTIVE_PROFILE", DEMO_TEST)
    r = app.app.test_client().post("/predict", json=FIXTURES["cases"][0]["payload"]).get_json()
    assert r["profile"] == "demo_rc" and r["profile_thresholds"]["peak_min_g"] == 1.2


# ── incident store ────────────────────────────────────────────────────────────
@pytest.fixture
def store():
    """Fresh in-memory store + app whose server profile the test can switch."""
    holder = {"profile": app.ACTIVE_PROFILE}
    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    flask_app = Flask("store_test")

    def classify(arrays):
        return app.run_inference(**arrays, profile=holder["profile"])

    flask_app.register_blueprint(events.create_blueprint(classify, app.parse_window, lambda: holder["profile"], eng))
    return flask_app.test_client(), holder


def case(name):
    return next(c for c in FIXTURES["cases"] if c["expected"]["severity_name"] == name)


def post(client, payload, **meta):
    r = client.post("/api/v1/events", json=dict(payload, **meta))
    assert r.status_code == 201, r.get_json()
    return r.get_json()


def test_event_stores_raw_window_and_both_profiles(store):
    client, _ = store
    payload = case("Moderate")["payload"]
    created = post(client, payload, event_type="trigger", classification_source="api", device_profile="production",
                   device_severity_class=1, device_metrics={"trigger_mag_g": 2.4})
    assert created["profile"] == "production" and created["server_profile"] == "production"
    assert created["raw_window_stored"] and created["excluded_from_performance"] is None
    ev = client.get(f"/api/v1/events/{created['id']}").get_json()
    row = ev["event"]
    assert ev["profile"] == "production"
    assert row["server_profile"] == "production" and row["device_profile"] == "production"
    for k in CH:
        assert np.allclose(row["raw_window"][k], payload[k])
    canon = json.dumps({k: [float(v) for v in payload[k]] for k in CH}, separators=(",", ":"), sort_keys=True)
    assert row["window_sha256"] == hashlib.sha256(canon.encode()).hexdigest()
    listed = client.get("/api/v1/events").get_json()
    assert listed["profile"] == "production" and "raw_window" not in listed["events"][0]
    assert listed["events"][0]["server_profile"] == "production"


def test_event_validation(store):
    client, _ = store
    w = case("Normal")["payload"]
    assert client.post("/api/v1/events", json=dict(w, event_type="bogus")).status_code == 400
    assert client.post("/api/v1/events", json={"event_type": "trigger"}).status_code == 400
    r = client.post("/api/v1/events", json=dict(w, event_type="rc_collection", ground_truth="impact_low"))
    assert r.status_code == 400 and "run_conditions" in r.get_json()["error"]
    r = client.post("/api/v1/events", json=dict(w, event_type="rc_collection", ground_truth="severe",
                                               run_conditions=dict(speed_kmh=5, barrier="padded", angle_deg=0, crush="none")))
    assert r.status_code == 400
    assert client.post("/api/v1/events", json={"event_type": "manual_panic"}).status_code == 201


def test_api_key_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("EVENTS_API_KEY", "k1")
    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    a = Flask("keyed")
    a.register_blueprint(events.create_blueprint(lambda arr: app.run_inference(**arr), app.parse_window,
                                                 lambda: app.ACTIVE_PROFILE, eng))
    c = a.test_client()
    assert c.post("/api/v1/events", json={"event_type": "manual_panic"}).status_code == 401
    assert c.post("/api/v1/events", json={"event_type": "manual_panic"}, headers={"X-API-Key": "k1"}).status_code == 201


def test_demo_profile_incidents_excluded_from_performance_statistics(store):
    client, holder = store
    rc_run = dict(speed_kmh=18, barrier="rigid", angle_deg=0, crush="none", cond="A")
    severe, moderate, normal = case("Severe")["payload"], case("Moderate")["payload"], case("Normal")["payload"]

    # eligible: production server + production device, real triggers with verified labels
    eligible = [
        (severe, "Severe"), (moderate, "Moderate"), (normal, "Normal"), (normal, "Moderate"),
    ]
    for payload, truth in eligible:
        post(client, payload, event_type="trigger", classification_source="api", device_profile="production",
             ground_truth=truth)
    baseline = client.get("/api/v1/stats").get_json()["performance"]
    assert baseline["included_n"] == 4

    # each of these is labelled so that including it would change recall/precision/F1
    post(client, normal, event_type="manual_panic", classification_source="none", ground_truth="Severe")
    post(client, normal, event_type="trigger", classification_source="local", device_profile="demo_rc",
         ground_truth="Severe")
    post(client, normal, event_type="rc_collection", classification_source="api", device_profile="production",
         ground_truth="impact_high", run_conditions=rc_run)
    holder["profile"] = DEMO_TEST                     # server switched to the demo profile
    demo_server = post(client, normal, event_type="trigger", classification_source="api",
                       device_profile="demo_rc", ground_truth="Severe")
    assert demo_server["server_profile"] == "demo_rc" and demo_server["profile"] == "demo_rc"
    assert demo_server["excluded_from_performance"] == "non_production_profile"
    post(client, normal, event_type="trigger", classification_source="api", device_profile="production",
         ground_truth="Moderate")                     # demo server, production device
    holder["profile"] = app.ACTIVE_PROFILE
    post(client, moderate, event_type="trigger", classification_source="api", device_profile="production")  # unlabelled

    stats = client.get("/api/v1/stats").get_json()
    perf = stats["performance"]
    print("\nperformance statistics after adding excluded rows:\n" + json.dumps(perf, indent=2))
    assert stats["profile"] == "production"
    assert perf["included_n"] == 4
    assert perf["excluded"] == {"manual_panic": 1, "non_production_profile": 3, "rc_collection": 1,
                                "no_verified_label": 1}
    for key in ("confusion_matrix", "per_class", "macro_f1", "crash_detection"):
        assert perf[key] == baseline[key], key


def test_exclusion_rule_table():
    base = dict(event_type="trigger", server_profile="production", device_profile="production")
    assert events.performance_exclusion_reason(base) is None
    assert events.performance_exclusion_reason(dict(base, device_profile=None)) is None
    assert events.performance_exclusion_reason(dict(base, event_type="manual_panic")) == "manual_panic"
    assert events.performance_exclusion_reason(dict(base, server_profile="demo_rc")) == "non_production_profile"
    assert events.performance_exclusion_reason(dict(base, device_profile="demo_rc")) == "non_production_profile"
    assert events.performance_exclusion_reason(dict(base, event_type="rc_collection")) == "rc_collection"


# ── firmware reads the same definition ────────────────────────────────────────
def test_firmware_header_is_generated_from_current_json():
    assert gen.HEADER.read_text(encoding="utf-8") == gen.render()


def test_firmware_filter_matches_server_filter():
    sos, zi, padlen = gen.filter_design()
    assert np.array_equal(sos, app._SOS)
    x = np.random.default_rng(1).normal(size=500)
    from scipy import signal
    assert padlen == 15 and np.allclose(signal.sosfiltfilt(sos, x, padlen=padlen), app._lp(x))


GXX = shutil.which("g++")


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    if not GXX:
        pytest.skip("g++ not available")
    d = tmp_path_factory.mktemp("gate")
    src = str(ROOT / "tests" / "firmware_gate_harness.cpp")
    prod, demo = d / "prod.exe", d / "demo.exe"
    subprocess.run([GXX, "-O2", "-std=c++17", "-o", str(prod), src], check=True)
    out = subprocess.run([GXX, "-O2", "-std=c++17", "-DTHRESHOLD_PROFILE_DEMO_RC", "-o", str(demo), src],
                         check=True, capture_output=True, text=True)
    return prod, demo, out.stderr


def test_firmware_profile_selection(harness):
    prod, demo, demo_warnings = harness
    p = subprocess.run([str(prod), "--profile"], capture_output=True, text=True).stdout.strip().split("|")
    assert p[:3] == ["production", "production", ""] and p[3:8] == ["2", "7", "40", "250", "0.959"]
    d = subprocess.run([str(demo), "--profile"], capture_output=True, text=True).stdout.strip().split("|")
    demo_def = profiles.load_definitions()["profiles"]["demo_rc"]
    if demo_def["status"] != "derived":
        assert d[0] == "production" and d[1] == "demo_rc" and "fell back to production" in d[2]
        assert "fell back to production" in demo_warnings
    else:
        assert d[0] == "demo_rc"


@pytest.mark.parametrize("prof", [app.ACTIVE_PROFILE, DEMO_TEST], ids=["production", "demo-like"])
def test_firmware_gate_matches_server_gate(harness, prof):
    prod_exe = harness[0]
    rng = np.random.default_rng(3)
    wins = [np.asarray([c["payload"][k] for k in ("ax", "ay", "az")], np.float32) for c in FIXTURES["cases"]]
    for _ in range(150):
        w = np.zeros((3, 500), np.float32)
        w[2] = 1
        w += rng.normal(0, .03, (3, 500)).astype(np.float32)
        s, n = rng.integers(0, 495), rng.integers(1, 30)
        w[rng.integers(0, 3), s:s + n] += rng.uniform(0.3, 12)
        wins.append(w)
    args = [] if prof is app.ACTIVE_PROFILE else [str(v) for v in (prof.peak_min_g, prof.peak_max_g, prof.dur_min_ms,
                                                                   prof.dur_max_ms, prof.severe_impulse_gs)]
    inp = "\n".join(" ".join(f"{v:.9g}" for v in w.ravel()) for w in wins)
    lines = subprocess.run([str(prod_exe), *args], input=inp, capture_output=True, text=True).stdout.splitlines()
    assert len(lines) == len(wins)
    for w, line in zip(wins, lines):
        pk, ex, im, sg, cl = line.split()
        peak, exc, imp, sig = profiles.crash_signature(*(app._lp(w[i].astype(float)) for i in range(3)), prof)
        assert abs(peak - float(pk)) < 1e-4 and exc == float(ex) and abs(imp - float(im)) < 1e-4
        assert sig == bool(int(sg)) and profiles.grade_by_impulse(imp, sig, prof) == int(cl)


# ── analysis script guards ────────────────────────────────────────────────────
def _run_analysis(tmp_path, rows, *extra):
    from rc_synthetic import write_jsonl
    data = tmp_path / "events.jsonl"
    write_jsonl(data, rows)
    return subprocess.run([sys.executable, str(ROOT / "analysis" / "rc_demo_profile.py"), "--input", str(data),
                           "--out", str(tmp_path / "out"), *extra], capture_output=True, text=True, cwd=ROOT)


def test_analysis_runs_end_to_end_and_refuses_synthetic_profile(tmp_path):
    from rc_synthetic import make_events
    before = profiles.PROFILES_FILE.read_bytes()
    r = _run_analysis(tmp_path, make_events(30), "--write-profile")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "REFUSING --write-profile" in r.stdout and "synthetic" in r.stdout
    assert profiles.PROFILES_FILE.read_bytes() == before
    out = json.loads((tmp_path / "out" / "demo_rc_derivation.json").read_text())
    assert out["dataset"]["synthetic"] is True
    for k in profiles.THRESHOLD_KEYS:
        assert isinstance(out["thresholds"][k], float)
    assert np.array(out["cross_validated"]["server"]["matrix"]).sum() == 90 + 12
    assert out["crush_structure"] and out["crush_structure"][0]["structure"] == "foam"
    assert len(out["figures"]) == 4


def test_analysis_refuses_underpowered_dataset(tmp_path):
    from rc_synthetic import make_events
    rows = [dict(r, synthetic=False) for r in make_events(8, crush_pairs=0)]
    before = profiles.PROFILES_FILE.read_bytes()
    r = _run_analysis(tmp_path, rows, "--write-profile")
    assert r.returncode == 3 and "below 30 events" in r.stdout
    assert profiles.PROFILES_FILE.read_bytes() == before
