"""
Phase 3 staged-API smoke test. Posts synthetic windows to a running deploy/app.py
and checks the deployed taxonomy behaves. Covers the five brief cases plus a
crash from the MISSED-19% population (a sub-40 ms pulse the signature vetoes) —
included to make the system recall ceiling explicit, not hidden.

Usage:  python smoke_test.py [http://localhost:5000]
"""
import sys, json, numpy as np, urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000") + "/predict"
RNG = np.random.default_rng(0)

def base(m=0.05):
    return (RNG.normal(0, m, 500), RNG.normal(0, m, 500), 1.0 + RNG.normal(0, m, 500), RNG.normal(0, 2, 500))
def spike(peak=12, w=5):
    ax, ay, az, gx = base(); ax[250:250+w] += peak; return ax, ay, az, gx
def crash(peak=4, dur_ms=90):
    ax, ay, az, gx = base(0.1); wn = int(dur_ms/10); t = np.arange(500)
    env = np.exp(-0.5*((t-250)/(wn/2))**2); ax += peak*env; ay += 0.4*peak*env; gx += 120*env
    return ax, ay, az, gx
def sustained(peak=9):
    ax, ay, az, gx = base(0.1); ax[150:400] += peak; return ax, ay, az, gx

def post(sig):
    ax, ay, az, gx = sig
    body = json.dumps({"ax": ax.tolist(), "ay": ay.tolist(), "az": az.tolist(),
                       "gx": gx.tolist(), "gy": gx.tolist(), "gz": gx.tolist()}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=20).read())

CASES = [
    ("normal driving",          base(),            "Normal",     "must be Normal"),
    ("brief spike (drop) 12 g", spike(12, 5),      "Normal",     "must be Normal"),
    ("sustained 9 g maneuver",  sustained(9),      "Normal",     "must be Normal"),
    ("real crash 4 g / 90 ms",  crash(4, 90),      "crash",      "must be Moderate/Severe"),
    ("real crash 6 g / 150 ms", crash(6, 150),     "crash",      "must be Moderate/Severe"),
    ("MISSED-19% crash 3 g/30 ms", crash(3, 30),   "known_miss", "sub-40ms: signature vetoes -> Normal (recall-ceiling case)"),
]

if __name__ == "__main__":
    for name, sig, expect, note in CASES:
        r = post(sig); sev = r["severity_name"]
        ok = {"Normal": sev == "Normal", "crash": sev in ("Moderate", "Severe"),
              "known_miss": sev == "Normal"}[expect]
        tag = "PASS" if ok else "FAIL"
        if expect == "known_miss":
            tag = "PASS(known-miss)" if sev == "Normal" else "UNEXPECTED"
        print(f"[{tag}] {name:26s} -> {sev:9s} p_crash={r['p_crash']:.2f} "
              f"sig={r['crash_signature']['signature_match']}  ({note})")
