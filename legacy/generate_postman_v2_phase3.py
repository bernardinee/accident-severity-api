"""
Generate a Postman collection for the Phase 3 v2 API (signature+impulse taxonomy).

The May-2026 collections (Postman_Tests_FINAL.json et al.) encode the OLD
peak-threshold rule as their expectations -- 12 g / 22 g / 35 g -> Severe. Under
v2 those inputs are correctly Normal (>7 g is a maneuver or artifact, not a
crash), so the old suites report the fix as a failure. This regenerates the
suite against v2 semantics, reusing the exact waveforms from smoke_test.py so
the Postman results and the smoke test agree case for case.

Usage:  python generate_postman_v2.py
Output: Postman_v2_Tests.json  (import into Postman, then Run collection)
"""
import json
import numpy as np

PROD = "https://accident-severity-api-production.up.railway.app"
RNG = np.random.default_rng(0)          # same seed/order as smoke_test.py


def base(m=0.05):
    return (RNG.normal(0, m, 500), RNG.normal(0, m, 500),
            1.0 + RNG.normal(0, m, 500), RNG.normal(0, 2, 500))


def spike(peak=12, w=5):
    ax, ay, az, gx = base(); ax[250:250 + w] += peak; return ax, ay, az, gx


def crash(peak=4, dur_ms=90):
    ax, ay, az, gx = base(0.1); wn = int(dur_ms / 10); t = np.arange(500)
    env = np.exp(-0.5 * ((t - 250) / (wn / 2)) ** 2)
    ax += peak * env; ay += 0.4 * peak * env; gx += 120 * env
    return ax, ay, az, gx


def sustained(peak=9):
    ax, ay, az, gx = base(0.1); ax[150:400] += peak; return ax, ay, az, gx


def body(sig):
    ax, ay, az, gx = sig
    r = lambda a: [round(float(v), 5) for v in a]
    return {"ax": r(ax), "ay": r(ay), "az": r(az),
            "gx": r(gx), "gy": r(gx), "gz": r(gx)}


# Generated in the same order as smoke_test.py's CASES so the RNG stream — and
# therefore every waveform — is identical to the run we validated.
CASES = [
    ("Normal driving", base(), "Normal", False,
     "Baseline: no crash signature, model agrees."),
    ("Brief 12 g spike (dropped device)", spike(12, 5), "Normal", False,
     "REGRESSION GUARD. The v1 API escalated this to Severe on peak alone. "
     "v2 returns Normal: >7 g with a ~50 ms transient is an artifact, not a crash."),
    ("Sustained 9 g maneuver", sustained(9), "Normal", False,
     "Long high-g event: above the 7 g crash ceiling, no crash signature."),
    ("Real crash 4 g / 90 ms", crash(4, 90), "crash", True,
     "In-band crash: 2-7 g peak, 40-250 ms transient."),
    ("Real crash 6 g / 150 ms", crash(6, 150), "crash", True,
     "In-band crash near the upper bound."),
    ("Sub-40 ms crash 3 g / 30 ms (known miss)", crash(3, 30), "known_miss", False,
     "DOCUMENTED LIMITATION, not a bug. The 40 ms duration floor vetoes this, "
     "which is what caps system recall at 81%. Set SIG_TRANSIENT_MIN_MS=20 on "
     "the service and this case flips to a crash (recall ~95%)."),
]

COMMON = [
    'pm.test("200 OK", () => pm.response.to.have.status(200));',
    'const d = pm.response.json();',
    'pm.test("v2 response shape", () => {',
    '    pm.expect(d).to.have.property("severity_name");',
    '    pm.expect(d).to.have.property("p_crash");',
    '    pm.expect(d).to.have.property("crash_signature");',
    '    pm.expect(d).to.have.property("label_source");',
    '});',
]

items = [{
    "name": "0. Health check",
    "request": {"method": "GET", "header": [], "url": {"raw": "{{base_url}}/health",
                "host": ["{{base_url}}"], "path": ["health"]}},
    "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": [
        'pm.test("200 OK", () => pm.response.to.have.status(200));',
        'const d = pm.response.json();',
        'pm.test("healthy", () => pm.expect(d.status).to.eql("healthy"));',
        'pm.test("v2 taxonomy is live", () =>',
        '    pm.expect(d.taxonomy).to.eql("signature+impulse (v2)"));',
        'pm.test("v2 model is loaded", () =>',
        '    pm.expect(d.model).to.eql("phase2_xgboost_calibrated"));',
        'pm.test("25 features", () => pm.expect(d.n_features).to.eql(25));',
        'pm.test("calibrated threshold 0.396", () =>',
        '    pm.expect(d.crash_alert_threshold).to.be.closeTo(0.396, 0.001));',
    ]}}],
}]

for i, (name, sig, expect, want_sig, note) in enumerate(CASES, start=1):
    if expect == "Normal":
        checks = [
            f'pm.test("classified Normal", () => pm.expect(d.severity_name).to.eql("Normal"));',
            'pm.test("no crash signature matched", () =>',
            '    pm.expect(d.crash_signature.signature_match).to.eql(false));',
        ]
    elif expect == "crash":
        checks = [
            'pm.test("classified as a crash", () =>',
            '    pm.expect(d.severity_name).to.be.oneOf(["Moderate", "Severe"]));',
            'pm.test("crash signature matched", () =>',
            '    pm.expect(d.crash_signature.signature_match).to.eql(true));',
            'pm.test("model agrees, P(crash) over threshold", () =>',
            '    pm.expect(d.p_crash).to.be.above(0.396));',
            'pm.test("alert raised", () => pm.expect(d.accident_confirmed).to.eql(true));',
        ]
    else:  # known_miss
        checks = [
            'pm.test("known miss: vetoed to Normal by the 40 ms floor", () =>',
            '    pm.expect(d.severity_name).to.eql("Normal"));',
            'pm.test("excursion is under the 40 ms floor", () =>',
            '    pm.expect(d.crash_signature.excursion_ms).to.be.below(40));',
            'console.log("Recall-ceiling case. Set SIG_TRANSIENT_MIN_MS=20 to catch it.");',
        ]
    items.append({
        "name": f"{i}. {name}",
        "request": {
            "method": "POST",
            "header": [{"key": "Content-Type", "value": "application/json"}],
            "body": {"mode": "raw", "raw": json.dumps(body(sig)),
                     "options": {"raw": {"language": "json"}}},
            "url": {"raw": "{{base_url}}/predict", "host": ["{{base_url}}"],
                    "path": ["predict"]},
            "description": note,
        },
        "event": [{"listen": "test", "script": {"type": "text/javascript",
                   "exec": COMMON + checks}}],
    })

# Contract / validation cases
bad = body(base())
bad.pop("gz")
items.append({
    "name": "7. Contract: missing field rejected",
    "request": {"method": "POST",
                "header": [{"key": "Content-Type", "value": "application/json"}],
                "body": {"mode": "raw", "raw": json.dumps(bad),
                         "options": {"raw": {"language": "json"}}},
                "url": {"raw": "{{base_url}}/predict", "host": ["{{base_url}}"],
                        "path": ["predict"]},
                "description": "gz omitted -> 400 with an explanatory error."},
    "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": [
        'pm.test("400 Bad Request", () => pm.response.to.have.status(400));',
        'pm.test("names the missing field", () =>',
        '    pm.expect(pm.response.json().error).to.include("gz"));',
    ]}}],
})

short = {k: v[:100] for k, v in body(base()).items()}
items.append({
    "name": "8. Contract: wrong sample count rejected",
    "request": {"method": "POST",
                "header": [{"key": "Content-Type", "value": "application/json"}],
                "body": {"mode": "raw", "raw": json.dumps(short),
                         "options": {"raw": {"language": "json"}}},
                "url": {"raw": "{{base_url}}/predict", "host": ["{{base_url}}"],
                        "path": ["predict"]},
                "description": "100 samples instead of 500 -> 400."},
    "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": [
        'pm.test("400 Bad Request", () => pm.response.to.have.status(400));',
        'pm.test("states the required window size", () =>',
        '    pm.expect(pm.response.json().error).to.include("500"));',
    ]}}],
})

collection = {
    "info": {
        "name": "Accident Severity API - v2 (signature+impulse)",
        "description": (
            "Phase 3 v2 test suite. Crash = a 2-7 g resultant transient lasting "
            "40-250 ms, graded by impulse - NOT peak height. Supersedes the May-2026 "
            "collections, whose expectations encode the retired peak-threshold rule "
            "and therefore fail against correct v2 behaviour.\n\n"
            "Set the base_url variable to target an environment; it defaults to "
            "production. Run with the Collection Runner."),
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
    },
    "variable": [{"key": "base_url", "value": PROD, "type": "string"}],
    "item": items,
}

with open("Postman_v2_Tests.json", "w", encoding="utf-8") as f:
    json.dump(collection, f, indent=1)

print(f"Wrote Postman_v2_Tests.json - {len(items)} requests")
