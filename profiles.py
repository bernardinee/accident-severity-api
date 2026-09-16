"""Threshold profiles for the post-model crash-signature gate.

Two named threshold sets live in artifacts/threshold_profiles.json:
  production - the thesis taxonomy (VZCrash-derived). Always the default.
  demo_rc    - scaled demonstration set for the RC-car demonstrator, written only
               by analysis/rc_demo_profile.py from measured impacts.

Only the gate differs between profiles; the model and its calibration do not.
Resolution never lands on a non-production profile by accident: an unset,
unrecognised or not-yet-derived profile resolves to production, and the reason
is carried in `note` so every response shows what happened.
"""
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

PROFILES_FILE = Path(__file__).parent / "artifacts" / "threshold_profiles.json"
PROFILE_ENV = "THRESHOLD_PROFILE"
PRODUCTION = "production"
THRESHOLD_KEYS = ("peak_min_g", "peak_max_g", "dur_min_ms", "dur_max_ms", "severe_impulse_gs")
FS = 100


@dataclass(frozen=True)
class Profile:
    name: str
    peak_min_g: float
    peak_max_g: float
    dur_min_ms: float
    dur_max_ms: float
    severe_impulse_gs: float
    severity_grading: str          # "model_ratio" (P(Severe|crash)) or "impulse"
    requested: Optional[str]       # raw configured value, None if unset
    note: Optional[str]            # why the active profile differs from `requested`

    def thresholds(self):
        d = {k: getattr(self, k) for k in THRESHOLD_KEYS}
        d["run_level_g"] = self.peak_min_g
        d["severity_grading"] = self.severity_grading
        return d

    def response_fields(self):
        """Fields added to every /predict and /api/v1/events response."""
        return {"profile": self.name, "profile_thresholds": self.thresholds(),
                "profile_requested": self.requested, "profile_note": self.note}

    def describe(self):
        return asdict(self)


def load_definitions(path=PROFILES_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_usable(p):
    return p.get("status") == "derived" and all(isinstance(p.get(k), (int, float)) for k in THRESHOLD_KEYS)


def resolve(requested=None, path=PROFILES_FILE, env=None):
    """Return the active Profile. Anything other than a usable, named profile -> production."""
    env = os.environ if env is None else env
    defs = load_definitions(path)["profiles"]
    prod = defs[PRODUCTION]
    if not _is_usable(prod):
        raise RuntimeError(f"{path}: production profile is incomplete; refusing to start")
    raw = requested if requested is not None else env.get(PROFILE_ENV)
    name = (raw or "").strip().lower()
    note = None
    if not name:
        name = PRODUCTION
    elif name not in defs:
        note = f"unrecognised profile {raw!r}; fell back to production"
        name = PRODUCTION
    elif not _is_usable(defs[name]):
        note = f"profile {name!r} is {defs[name].get('status')!r} (thresholds not derived from measured data); fell back to production"
        name = PRODUCTION
    p = defs[name]
    vals = {k: float(p[k]) for k in THRESHOLD_KEYS}
    if name == PRODUCTION and env.get("SIG_TRANSIENT_MIN_MS"):
        # pre-existing operator override of the production duration floor (Phase 3 Task 2)
        vals["dur_min_ms"] = float(env["SIG_TRANSIENT_MIN_MS"])
    return Profile(name=name, severity_grading=p["severity_grading"], requested=raw or None, note=note, **vals)


def crash_signature(ax, ay, az, profile):
    """Physics gate on (filtered) g-unit axes: peak_g, longest run >= run level (ms), impulse (g·s), match."""
    import numpy as np
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    peak = float(mag.max())
    above = mag >= profile.peak_min_g
    longest = run = 0
    for a in above:
        run = run + 1 if a else 0
        longest = max(longest, run)
    longest_ms = longest * 1000.0 / FS
    impulse = float(np.sum(mag[above] - 1.0) / FS) if above.any() else 0.0
    is_sig = (profile.peak_min_g <= peak < profile.peak_max_g
              and profile.dur_min_ms <= longest_ms <= profile.dur_max_ms)
    return peak, longest_ms, impulse, bool(is_sig)


def grade_by_impulse(impulse, is_sig, profile):
    """Model-free classification (firmware classifyLocally): 0 Normal, 1 Moderate, 2 Severe."""
    if not is_sig:
        return 0
    return 2 if impulse >= profile.severe_impulse_gs else 1
