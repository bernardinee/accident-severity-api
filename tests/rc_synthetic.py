"""SYNTHETIC RC-car events for exercising analysis/rc_demo_profile.py in tests.

Every row is flagged "synthetic": true, and the analysis script refuses to write a
profile from such data. These numbers are NOT measurements and must never
appear in DEMO_PROFILE.md.
"""
import json

import numpy as np

N = 500


def _window(rng, pulses=(), bump=None):
    ax = rng.normal(0, 0.04, N); ay = rng.normal(0, 0.04, N); az = 1 + rng.normal(0, 0.04, N)
    for start, width, amp in pulses:
        t = np.arange(width)
        ax[start:start + width] -= amp * np.sin(np.pi * (t + 0.5) / width)
    if bump:
        s, w, a = bump
        az[s:s + w] += a
    g = rng.normal(0, 2, (3, N))
    return dict(ax=ax.tolist(), ay=ay.tolist(), az=az.tolist(), gx=g[0].tolist(), gy=g[1].tolist(), gz=g[2].tolist())


def make_events(n_per_class=30, seed=0, crush_pairs=6):
    rng = np.random.default_rng(seed)
    rows, i = [], 0

    def add(label, window, run):
        nonlocal i
        i += 1
        rows.append(dict(id=i, event_type="rc_collection", ground_truth=label, synthetic=True,
                         run_conditions=run, raw_window=window))

    for _ in range(n_per_class):
        add("normal", _window(rng, bump=(250, int(rng.integers(1, 4)), float(rng.uniform(0.3, 1.5)))),
            dict(speed_kmh=rng.uniform(3, 15), barrier="none", angle_deg=0, crush="none", cond=""))
        add("impact_low", _window(rng, [(250, int(rng.integers(3, 6)), float(rng.uniform(3, 5)))]),
            dict(speed_kmh=rng.uniform(4, 8), barrier="padded", angle_deg=0, crush="none", cond=""))
        add("impact_high", _window(rng, [(250, int(rng.integers(4, 8)), float(rng.uniform(6, 9)))]),
            dict(speed_kmh=rng.uniform(15, 20), barrier="rigid", angle_deg=0, crush="none", cond=""))
    for k in range(crush_pairs):
        add("impact_high", _window(rng, [(250, 4, 8.0)]),
            dict(speed_kmh=18, barrier="rigid", angle_deg=0, crush="none", cond="C1"))
        add("impact_high", _window(rng, [(250, 9, 4.5)]),
            dict(speed_kmh=18, barrier="rigid", angle_deg=0, crush="foam", cond="C1"))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
