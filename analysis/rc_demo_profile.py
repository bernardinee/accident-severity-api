"""Derive the demo_rc threshold profile from measured RC-car events.

Input: data/rc_collection/events.jsonl, exported from /api/v1/events with
tools/export_events.py. Each row holds the raw 500-sample window, a ground-truth
label (normal | impact_low | impact_high) and the run conditions logged at
collection time.

Every metric is computed on the SERVE PATH (app.normalize_to_g, then app._lp's
20 Hz zero-phase Butterworth), because that is the signal the gate sees. Raw
100 Hz metrics are reported alongside for the Phase 2 side-by-side and the crush
comparison.

Derivation rules are fixed here, before any data is seen (DEMO_PROFILE.md §A.5):
  peak band       floor and ceiling each maximise Youden's J (impacts in band minus
                  handling events in band); a tied interval resolves to its midpoint
                  (max-margin). Ceiling search stops at the ±16 g sensor range.
  duration window floor/ceiling sit at the same positions in the impact-duration
                  distribution as production's 40/250 ms sit in VZCrash's:
                  17.2% of crash events below the floor, 1.8% above the ceiling.
  severe impulse  J-optimal split between impact_low and impact_high impulse
                  (max-margin midpoint); reported as delta-v = impulse x 9.80665.
Evaluation: 3x3 confusion in-sample AND 5-fold stratified cross-validation
(thresholds re-derived on each training fold), gate-only (firmware
classifyLocally) and full server decision (model P(crash) vote AND gate).

    python analysis/rc_demo_profile.py                       # analyse, write analysis/out/
    python analysis/rc_demo_profile.py --write-profile       # also write demo_rc into artifacts/
    python analysis/rc_demo_profile.py --vzcrash ../phase2/results/vzcrash_windows.csv
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("EVENTS_DATABASE_URL", "sqlite://")      # importing app must not create a store
import app  # noqa: E402
import profiles  # noqa: E402

G = 9.80665
FS = 100
LABELS = ("normal", "impact_low", "impact_high")
TRUE_CLASS = {"normal": 0, "impact_low": 1, "impact_high": 2}
MIN_PER_CLASS = 30
# Production's percentile positions in VZCrash (27,711 crash events; phase2/results/vzcrash_windows.csv)
PROD_FRAC_BELOW_FLOOR, PROD_FRAC_ABOVE_CEIL = 0.1722, 0.0178
PEAK_GRID = np.round(np.arange(1.0, 16.0 + 1e-9, 0.05), 2)
SENSOR_MAX_G = 16.0
# figure palette: validated categorical slots 1-4 (dataviz reference palette, light surface)
COLORS = {"normal": "#2a78d6", "impact_low": "#eb6834", "impact_high": "#1baf7a", "vzcrash": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ── loading ───────────────────────────────────────────────────────────────────
def load(path):
    rows, skipped = [], {"not_rc_collection": 0, "unlabelled": 0, "no_window": 0}
    raw = Path(path).read_bytes()
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event_type") != "rc_collection":
            skipped["not_rc_collection"] += 1
        elif r.get("ground_truth") not in LABELS:
            skipped["unlabelled"] += 1
        elif not r.get("raw_window"):
            skipped["no_window"] += 1
        else:
            rows.append(r)
    return rows, skipped, hashlib.sha256(raw).hexdigest()


def longest_run(mask):
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best * 1000.0 / FS


def measure(r):
    w = r["raw_window"]
    ax, ay, az = (np.asarray(w[k], float) for k in ("ax", "ay", "az"))
    nx, ny, nz, scale = app.normalize_to_g(ax, ay, az)
    raw_mag = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    fx, fy, fz = app._lp(nx), app._lp(ny), app._lp(nz)
    mag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    jerk = np.abs(np.diff(raw_mag, prepend=raw_mag[0])) * FS
    run = r.get("run_conditions") or {}
    pred = app.run_inference(*(np.asarray(w[k], float) for k in ("ax", "ay", "az", "gx", "gy", "gz")),
                             profile=profiles.resolve(requested="production"))
    return dict(
        id=r.get("id"), label=r["ground_truth"], synthetic=bool(r.get("synthetic")),
        speed_kmh=pd.to_numeric(run.get("speed_kmh"), errors="coerce"), barrier=run.get("barrier"),
        angle_deg=pd.to_numeric(run.get("angle_deg"), errors="coerce"), crush=(run.get("crush") or "none"),
        cond=run.get("cond") or "", unit_scale=scale,
        device_trigger=bool(((raw_mag >= 2.0) & (jerk >= 5.0)).any()),
        p_crash=pred["p_crash"],
        peak_raw=float(raw_mag.max()), peak=float(mag.max()),
        dur_raw_2g=longest_run(raw_mag >= 2.0), dur_2g=longest_run(mag >= 2.0),
        _mag=mag,
    )


def gate_metrics(df, run_level):
    """Serve-path excursion (ms) and impulse (g·s) above a run level, per window."""
    dur = np.array([longest_run(m >= run_level) for m in df["_mag"]])
    imp = np.array([float(np.sum(m[m >= run_level] - 1.0) / FS) for m in df["_mag"]])
    return dur, imp


# ── derivation ────────────────────────────────────────────────────────────────
def _tied_midpoint(grid, score):
    """Midpoint of the first contiguous run of grid points attaining the maximum score."""
    best = score.max()
    top = np.isclose(score, best)
    start = int(np.argmax(top))
    end = start
    while end + 1 < len(top) and top[end + 1]:
        end += 1
    return (grid[start] + grid[end]) / 2, float(best)


def _maxmargin(grid, score):
    mid, best = _tied_midpoint(grid, score)
    return round(float(np.round(mid / 0.05) * 0.05), 2), best


def derive(df):
    imp_mask = df.label != "normal"
    ip, npk = df.peak[imp_mask].to_numpy(), df.peak[~imp_mask].to_numpy()
    tpr = np.array([(ip >= t).mean() for t in PEAK_GRID])
    fpr = np.array([(npk >= t).mean() for t in PEAK_GRID]) if len(npk) else np.zeros(len(PEAK_GRID))
    lo, j_lo = _maxmargin(PEAK_GRID, tpr - fpr)
    upper = PEAK_GRID[PEAK_GRID > lo]
    j_hi = np.array([((ip >= lo) & (ip < t)).mean() - (((npk >= lo) & (npk < t)).mean() if len(npk) else 0)
                     for t in upper])
    hi, j_band = _maxmargin(upper, j_hi)
    hi = round(min(hi, SENSOR_MAX_G), 2)

    dur, imp = gate_metrics(df, lo)
    d_imp = dur[imp_mask.to_numpy()]
    cands = np.arange(10, 5010, 10)
    floor = max([v for v in cands if (d_imp < v).mean() <= PROD_FRAC_BELOW_FLOOR], default=10.0)
    ceil = min([v for v in cands if (d_imp > v).mean() <= PROD_FRAC_ABOVE_CEIL], default=float(cands[-1]))

    low = imp[(df.label == "impact_low").to_numpy()]
    high = imp[(df.label == "impact_high").to_numpy()]
    grid = np.unique(np.round(np.concatenate([low, high, [0.0]]), 4))
    grid = np.linspace(grid.min(), grid.max() + 1e-3, 2000)
    j_sev = np.array([(high >= t).mean() - (low >= t).mean() for t in grid])
    mid, best = _tied_midpoint(grid, j_sev)
    severe = float(np.round(mid, 3))

    in_band = (df.peak >= lo) & (df.peak < hi)
    return dict(
        peak_min_g=lo, peak_max_g=hi, dur_min_ms=float(floor), dur_max_ms=float(ceil), severe_impulse_gs=severe,
        youden_peak_floor=round(j_lo, 4), youden_peak_band=round(j_band, 4), youden_severe=round(float(best), 4),
        impacts_in_peak_band=round(float(in_band[imp_mask].mean()), 4),
        normals_in_peak_band=round(float(in_band[~imp_mask].mean()), 4) if (~imp_mask).any() else None,
        impacts_below_dur_floor=round(float((d_imp < floor).mean()), 4),
        impacts_above_dur_ceiling=round(float((d_imp > ceil).mean()), 4),
        severe_delta_v_ms=round(severe * G, 3), severe_delta_v_kmh=round(severe * G * 3.6, 2),
        p85_impact_impulse_gs=round(float(np.percentile(np.concatenate([low, high]), 85)), 3) if len(low) + len(high) else None,
    )


def as_profile(t, grading="impulse"):
    return profiles.Profile("demo_rc", t["peak_min_g"], t["peak_max_g"], t["dur_min_ms"], t["dur_max_ms"],
                            t["severe_impulse_gs"], grading, "demo_rc", None)


def predict(df, t, crash_threshold):
    """(gate-only, server) predicted class per window under thresholds t."""
    dur, imp = gate_metrics(df, t["peak_min_g"])
    sig = ((df.peak >= t["peak_min_g"]) & (df.peak < t["peak_max_g"]) &
           (dur >= t["dur_min_ms"]) & (dur <= t["dur_max_ms"])).to_numpy()
    graded = np.where(imp >= t["severe_impulse_gs"], 2, 1)
    trig = df.device_trigger.to_numpy()
    gate = np.where(trig & sig, graded, 0)
    server = np.where(trig & sig & (df.p_crash >= crash_threshold).to_numpy(), graded, 0)
    return gate, server


def confusion(y, p):
    cm = np.zeros((3, 3), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


def summarise_cm(cm):
    impacts = cm[1:].sum()
    normals = cm[0].sum()
    return dict(
        matrix=cm.tolist(),
        crash_recall=round(float(cm[1:, 1:].sum() / impacts), 4) if impacts else None,
        false_positive_rate=round(float(cm[0, 1:].sum() / normals), 4) if normals else None,
        impact_low_as_moderate=round(float(cm[1, 1] / cm[1].sum()), 4) if cm[1].sum() else None,
        impact_high_as_severe=round(float(cm[2, 2] / cm[2].sum()), 4) if cm[2].sum() else None,
        accuracy=round(float(np.trace(cm) / cm.sum()), 4) if cm.sum() else None,
    )


def cross_validate(df, folds=5, seed=42):
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(df), int)
    for lab in LABELS:
        idx = np.flatnonzero(df.label.to_numpy() == lab)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % folds
    y = df.label.map(TRUE_CLASS).to_numpy()
    cm_g, cm_s, per_fold = np.zeros((3, 3), int), np.zeros((3, 3), int), []
    for k in range(folds):
        tr, te = df[fold != k], df[fold == k]
        t = derive(tr)
        g, s = predict(te, t, app.CRASH_ALERT_THRESHOLD)
        cm_g += confusion(y[fold == k], g)
        cm_s += confusion(y[fold == k], s)
        per_fold.append({k2: t[k2] for k2 in profiles.THRESHOLD_KEYS})
    return summarise_cm(cm_g), summarise_cm(cm_s), per_fold


def describe(s):
    s = pd.Series(s).dropna()
    if s.empty:
        return dict(n=0)
    return dict(n=int(len(s)), median=round(float(s.median()), 3), p25=round(float(s.quantile(.25)), 3),
                p75=round(float(s.quantile(.75)), 3), iqr=round(float(s.quantile(.75) - s.quantile(.25)), 3),
                p5=round(float(s.quantile(.05)), 3), p95=round(float(s.quantile(.95)), 3))


def crush_comparison(df, lo):
    """Paired with/without crush structure, matched on the logged condition id."""
    from scipy.stats import mannwhitneyu
    dur_lo, _ = gate_metrics(df, lo) if lo else (np.full(len(df), np.nan), None)
    df = df.assign(dur_demo=dur_lo)
    out = []
    impacts = df[df.label != "normal"]
    for cond, g in impacts.groupby("cond"):
        base, fitted = g[g.crush == "none"], g[g.crush != "none"]
        if not cond or base.empty or fitted.empty:
            continue
        for structure, f in fitted.groupby("crush"):
            row = dict(cond=cond, structure=structure, n_without=len(base), n_with=len(f),
                       speed_kmh=float(g.speed_kmh.median()), barrier=str(g.barrier.mode().iat[0]))
            for m in ("peak_raw", "peak", "dur_raw_2g", "dur_2g", "dur_demo"):
                a, b = base[m].dropna(), f[m].dropna()
                row[f"{m}_without"] = round(float(a.median()), 3) if len(a) else None
                row[f"{m}_with"] = round(float(b.median()), 3) if len(b) else None
                row[f"{m}_p"] = (float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
                                 if len(a) > 1 and len(b) > 1 else None)
            row["reach_40ms_2g_without"] = round(float((base.dur_2g >= 40).mean()), 3)
            row["reach_40ms_2g_with"] = round(float((f.dur_2g >= 40).mean()), 3)
            out.append(row)
    return out


# ── figures (Phase 2 v1/v2/v3 equivalents, side by side with VZCrash) ─────────
def figures(df, t, outdir, vz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    outdir.mkdir(parents=True, exist_ok=True)
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(outdir / name, dpi=200)
        plt.close(fig)
        made.append(name)

    def hist(ax, data, bins, title, xlabel, lines=()):
        for lab, vals in data:
            ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=COLORS[lab], label=lab)
        for x, txt in lines:
            ax.axvline(x, color=INK2, linewidth=1, linestyle="--")
            ax.annotate(txt, (x, 1), xycoords=("data", "axes fraction"), fontsize=8, color=INK2,
                        xytext=(3, -12), textcoords="offset points")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("events")
        ax.legend()

    by = [(lab, df[df.label == lab]) for lab in LABELS]
    # v1: peak distribution — VZCrash (raw, Phase 2) | RC raw | RC serve path with derived band
    fig, axs = plt.subplots(1, 3 if vz is not None else 2, figsize=(12 if vz is not None else 8.5, 3.4))
    k = 0
    if vz is not None:
        hist(axs[0], [("vzcrash", vz.peak)], np.arange(0, 12.25, 0.25), "VZCrash crash events (raw 100 Hz)",
             "peak resultant (g)", [(2.0, "2 g"), (7.0, "7 g")])
        k = 1
    hist(axs[k], [(l, g.peak_raw) for l, g in by], np.arange(0, 28, 0.5), "RC car (raw 100 Hz)", "peak resultant (g)")
    hist(axs[k + 1], [(l, g.peak) for l, g in by], np.arange(0, 16.25, 0.25), "RC car (serve path, 20 Hz LP)",
         "peak resultant (g)", [(t["peak_min_g"], f"{t['peak_min_g']:.2f} g"), (t["peak_max_g"], f"{t['peak_max_g']:.2f} g")])
    save(fig, "v1_peak_hist_vzcrash_vs_rc.png")

    dur, imp = gate_metrics(df, t["peak_min_g"])
    df = df.assign(dur_demo=dur, imp_demo=imp)
    by = [(lab, df[df.label == lab]) for lab in LABELS]
    # v2: pulse duration — VZCrash >=2 g | RC >=2 g (production run level) | RC >= demo floor
    fig, axs = plt.subplots(1, 3 if vz is not None else 2, figsize=(12 if vz is not None else 8.5, 3.4))
    k = 0
    if vz is not None:
        hist(axs[0], [("vzcrash", vz.dur)], np.arange(0, 410, 10), "VZCrash: longest run >= 2 g",
             "duration (ms)", [(40, "40 ms"), (250, "250 ms")])
        k = 1
    bins = np.arange(0, max(200, float(np.nanmax(df.dur_raw_2g)) + 20), 10)
    hist(axs[k], [(l, g.dur_2g) for l, g in by], bins, "RC car: longest run >= 2 g (serve path)", "duration (ms)",
         [(40, "40 ms production floor")])
    hist(axs[k + 1], [(l, g.dur_demo) for l, g in by], bins,
         f"RC car: longest run >= {t['peak_min_g']:.2f} g (serve path)", "duration (ms)",
         [(t["dur_min_ms"], f"{t['dur_min_ms']:.0f} ms"), (t["dur_max_ms"], f"{t['dur_max_ms']:.0f} ms")])
    save(fig, "v2_duration_vzcrash_vs_rc.png")

    # v3: peak vs duration with the gate box
    fig, axs = plt.subplots(1, 2 if vz is not None else 1, figsize=(10 if vz is not None else 5.5, 4), squeeze=False)
    axs = axs[0]
    k = 0
    if vz is not None:
        s = vz.sample(min(len(vz), 4000), random_state=0)
        axs[0].scatter(s.dur, s.peak, s=8, color=COLORS["vzcrash"], edgecolor=SURFACE, linewidth=0.5, label="vzcrash")
        axs[0].add_patch(plt.Rectangle((40, 2), 210, 5, fill=False, edgecolor=INK2, linestyle="--", label="production gate"))
        axs[0].set(title="VZCrash (raw, 4,000-event sample)", xlabel="longest run >= 2 g (ms)", ylabel="peak (g)")
        axs[0].legend()
        k = 1
    for lab, g in by:
        axs[k].scatter(g.dur_demo, g.peak, s=40, color=COLORS[lab], edgecolor=SURFACE, linewidth=2, label=lab)
    axs[k].add_patch(plt.Rectangle((t["dur_min_ms"], t["peak_min_g"]), t["dur_max_ms"] - t["dur_min_ms"],
                                   t["peak_max_g"] - t["peak_min_g"], fill=False, edgecolor=INK2, linestyle="--",
                                   label="demo_rc gate"))
    axs[k].set(title="RC car (serve path)", xlabel=f"longest run >= {t['peak_min_g']:.2f} g (ms)", ylabel="peak (g)")
    axs[k].legend()
    save(fig, "v3_peak_vs_duration_vzcrash_vs_rc.png")

    # v4: impulse, low vs high energy, with severe cut
    fig, ax = plt.subplots(figsize=(6, 3.4))
    imps = df[df.label != "normal"]
    hist(ax, [(l, g.imp_demo) for l, g in by if l != "normal"],
         np.linspace(0, max(0.05, float(imps.imp_demo.max()) * 1.05), 30),
         "RC impacts: impulse above run level", "impulse (g·s)",
         [(t["severe_impulse_gs"], f"{t['severe_impulse_gs']:.3f} g·s = Δv {t['severe_impulse_gs'] * G:.2f} m/s")])
    save(fig, "v4_impulse_low_vs_high.png")
    return made


# ── report ────────────────────────────────────────────────────────────────────
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def cm_table(cm):
    names = ["Normal", "Moderate", "Severe"]
    return md_table(["true \\ predicted", *names],
                    [[lab, *row] for lab, row in zip(LABELS, cm["matrix"])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "rc_collection" / "events.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "analysis" / "out"))
    ap.add_argument("--vzcrash", default=None, help="phase2/results/vzcrash_windows.csv for side-by-side figures")
    ap.add_argument("--write-profile", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, skipped, sha = load(a.input)
    counts = {lab: sum(r["ground_truth"] == lab for r in rows) for lab in LABELS}
    print(f"dataset {a.input} sha256={sha[:16]} usable={len(rows)} by class={counts} skipped={skipped}")
    underpowered = [lab for lab, n in counts.items() if n < MIN_PER_CLASS]
    if not rows or any(counts[lab] == 0 for lab in LABELS):
        print("cannot derive: every class needs events (protocol minimum 30 each)")
        sys.exit(2)

    df = pd.DataFrame([measure(r) for r in rows])
    synthetic = bool(df.synthetic.any())
    t = derive(df)
    y = df.label.map(TRUE_CLASS).to_numpy()
    g_in, s_in = predict(df, t, app.CRASH_ALERT_THRESHOLD)
    prod = profiles.resolve(requested="production")
    prod_t = {k: getattr(prod, k) for k in profiles.THRESHOLD_KEYS}
    g_prod, s_prod = predict(df, prod_t, app.CRASH_ALERT_THRESHOLD)
    cv_gate, cv_server, cv_folds = cross_validate(df)

    dur_demo, imp_demo = gate_metrics(df, t["peak_min_g"])
    df = df.assign(dur_demo=dur_demo, imp_demo=imp_demo)
    dist = {lab: {m: describe(df.loc[df.label == lab, m]) for m in
                  ("peak_raw", "peak", "dur_raw_2g", "dur_2g", "dur_demo", "imp_demo", "p_crash", "speed_kmh")}
            for lab in LABELS}
    trig = {lab: round(float(df.loc[df.label == lab, "device_trigger"].mean()), 4) for lab in LABELS}
    model_vote = {lab: round(float((df.loc[df.label == lab, "p_crash"] >= app.CRASH_ALERT_THRESHOLD).mean()), 4)
                  for lab in LABELS}
    impacts = df[df.label != "normal"]
    dv_vs_speed = None
    if impacts.speed_kmh.notna().sum() >= 3:
        ratio = (impacts.imp_demo * G) / (impacts.speed_kmh / 3.6)
        dv_vs_speed = describe(ratio.replace([np.inf, -np.inf], np.nan))
    crush = crush_comparison(df, t["peak_min_g"])

    vz = None
    if a.vzcrash and Path(a.vzcrash).exists():
        w = pd.read_csv(a.vzcrash, usecols=["recording_id", "provenance", "peak_g", "longest_excursion_ms"])
        vz = (w[w.provenance == "CRASH"].groupby("recording_id")
              .agg(peak=("peak_g", "max"), dur=("longest_excursion_ms", "max")).reset_index())
    figs = figures(df.drop(columns=["dur_demo", "imp_demo"]), t, out / "figures", vz)

    result = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset=dict(path=str(Path(a.input).relative_to(ROOT)) if Path(a.input).is_relative_to(ROOT) else a.input,
                     sha256=sha, n_by_class=counts, skipped=skipped, synthetic=synthetic,
                     underpowered_classes=underpowered),
        thresholds=t, distributions=dist, device_trigger_rate=trig,
        model_crash_vote_rate=dict(threshold=app.CRASH_ALERT_THRESHOLD, by_class=model_vote),
        measured_delta_v_over_approach_speed=dv_vs_speed,
        in_sample=dict(gate_only=summarise_cm(confusion(y, g_in)), server=summarise_cm(confusion(y, s_in))),
        cross_validated=dict(gate_only=cv_gate, server=cv_server, fold_thresholds=cv_folds),
        production_profile_on_rc=dict(gate_only=summarise_cm(confusion(y, g_prod)), server=summarise_cm(confusion(y, s_prod))),
        crush_structure=crush, figures=figs,
    )
    (out / "demo_rc_derivation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    rep = [f"# demo_rc derivation — generated {result['generated_at']}", "",
           f"Dataset `{result['dataset']['path']}` sha256 `{sha}`; events by class {counts}; skipped {skipped}.",
           "**SYNTHETIC TEST DATA — NOT A MEASUREMENT.**" if synthetic else "",
           f"**UNDERPOWERED: {underpowered} below {MIN_PER_CLASS} events.**" if underpowered else "", "",
           "## Distributions (serve path unless marked raw)", ""]
    hdr = ["class", "metric", "n", "median", "IQR (p25–p75)", "p5", "p95"]
    trows = []
    names = dict(peak_raw="peak raw (g)", peak="peak (g)", dur_raw_2g="run ≥2 g raw (ms)", dur_2g="run ≥2 g (ms)",
                 dur_demo=f"run ≥{t['peak_min_g']:.2f} g (ms)", imp_demo="impulse (g·s)", p_crash="model P(crash)",
                 speed_kmh="logged speed (km/h)")
    for lab in LABELS:
        for m, d in dist[lab].items():
            if d.get("n"):
                trows.append([lab, names[m], d["n"], d["median"], f"{d['p25']}–{d['p75']}", d["p5"], d["p95"]])
    rep += [md_table(hdr, trows), "", "## Derived thresholds", "",
            md_table(["quantity", "value"], [[k, v] for k, v in t.items()]), "",
            f"Device trigger fired: {trig}. Model P(crash) ≥ {app.CRASH_ALERT_THRESHOLD}: {model_vote}.", "",
            "## Confusion matrices", "",
            "### In-sample, gate only (firmware classifyLocally)", cm_table(result["in_sample"]["gate_only"]),
            json.dumps({k: v for k, v in result["in_sample"]["gate_only"].items() if k != "matrix"}), "",
            "### In-sample, server (model vote AND gate)", cm_table(result["in_sample"]["server"]),
            json.dumps({k: v for k, v in result["in_sample"]["server"].items() if k != "matrix"}), "",
            "### 5-fold cross-validated, gate only", cm_table(cv_gate),
            json.dumps({k: v for k, v in cv_gate.items() if k != "matrix"}), "",
            "### 5-fold cross-validated, server", cm_table(cv_server),
            json.dumps({k: v for k, v in cv_server.items() if k != "matrix"}), "",
            "### Production profile applied to the same RC events (server)", cm_table(result["production_profile_on_rc"]["server"]), "",
            "## Crush structure (matched on condition id)", ""]
    if crush:
        keys = list(crush[0].keys())
        rep.append(md_table(keys, [[c[k] if not isinstance(c[k], float) else round(c[k], 4) for k in keys] for c in crush]))
    else:
        rep.append("**Not measured**: no condition id has both crush=none and a fitted structure.")
    rep += ["", "## Figures", "", *[f"![{f}](figures/{f})" for f in figs], ""]
    (out / "demo_rc_report.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"wrote {out / 'demo_rc_derivation.json'} and {out / 'demo_rc_report.md'}")
    print(f"thresholds {t}")
    print(f"cross-validated server: recall {cv_server['crash_recall']} FPR {cv_server['false_positive_rate']}")

    if a.write_profile:
        refusals = []
        if synthetic:
            refusals.append("dataset contains synthetic rows")
        if underpowered:
            refusals.append(f"classes below {MIN_PER_CLASS} events: {underpowered}")
        if not np.isfinite(t["severe_impulse_gs"]):
            refusals.append("severe impulse undefined")
        if refusals:
            print("REFUSING --write-profile: " + "; ".join(refusals))
            sys.exit(3)
        path = profiles.PROFILES_FILE
        defs = json.loads(path.read_text(encoding="utf-8"))
        demo = defs["profiles"]["demo_rc"]
        demo.update({k: t[k] for k in profiles.THRESHOLD_KEYS})
        demo["status"] = "derived"
        demo["provenance"] = dict(
            script="analysis/rc_demo_profile.py", derived_at=result["generated_at"], dataset_sha256=sha,
            n_by_class=counts, impacts_in_peak_band=t["impacts_in_peak_band"],
            normals_in_peak_band=t["normals_in_peak_band"],
            duration_rule=f"floor/ceiling at production's VZCrash positions: {PROD_FRAC_BELOW_FLOOR:.1%} below, "
                          f"{PROD_FRAC_ABOVE_CEIL:.1%} above",
            severe_rule="Youden-optimal impact_low/impact_high split, max-margin midpoint",
            severe_delta_v_ms=t["severe_delta_v_ms"],
            cross_validated_server=dict(crash_recall=cv_server["crash_recall"], fpr=cv_server["false_positive_rate"]))
        path.write_text(json.dumps(defs, indent=2) + "\n", encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "tools" / "gen_firmware_profiles.py")], check=True)
        print(f"demo_rc written to {path} and firmware header regenerated")


if __name__ == "__main__":
    main()
