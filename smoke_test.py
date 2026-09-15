"""
Smoke test: the six designed windows in tests/fixtures.json must return their
intended class — Normal driving, dropped-device spike, sustained 9 g, sub-40 ms
pulse (all Normal), a single ~4 g / 90 ms impact (Moderate) and a two-impact
high-impulse collision (Severe). For the full suite run test_api.py.

Usage:  python smoke_test.py [http://localhost:5000]
"""
import json, sys, urllib.request
from pathlib import Path

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000").rstrip("/")
fx = json.load(open(Path(__file__).parent / "tests" / "fixtures.json"))
failed = 0
for c in (c for c in fx["cases"] if c["kind"] == "synthetic"):
    req = urllib.request.Request(BASE + "/predict", json.dumps(c["payload"]).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    ok = r["severity_name"] == c["true_label"]
    failed += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {c['name']:52s} -> {r['severity_name']:8s} (want {c['true_label']:8s}) "
          f"p_crash={r['p_crash']:.2f} sig={r['crash_signature']['signature_match']} src={r['label_source']}")
sys.exit(1 if failed else 0)
