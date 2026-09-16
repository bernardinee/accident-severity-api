"""Incident store: /api/v1/events and /api/v1/stats.

Every incident row keeps the raw 500-sample window it was classified from, the
server's classification under the server's active threshold profile, the
device's own classification and compiled profile, and (for RC-car collection
runs) the ground-truth label and run conditions.

Performance statistics use exactly one eligibility rule,
performance_exclusion_reason(). A row counts only if it is a real triggered
incident classified under the production profile on BOTH server and device.
manual_panic, rc_collection and any non-production-profile row are excluded,
so demonstrator data can never reach a recall, precision or F1 figure.

Storage: EVENTS_DATABASE_URL (SQLAlchemy URL). Default sqlite:///data/events.db.
On Railway the container filesystem is wiped on every deploy, so production must
point this at Postgres or a mounted volume (see DEMO_PROFILE.md, section A.4).
"""
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, Response, jsonify, request
from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, Text,
                        create_engine, select)

import profiles

log = logging.getLogger(__name__)

CLASS_NAMES = ["Normal", "Moderate", "Severe"]
EVENT_TYPES = ("trigger", "manual_panic", "rc_collection")
CLASSIFICATION_SOURCES = ("api", "local", "none")
RC_LABELS = ("normal", "impact_low", "impact_high")
RC_RUN_KEYS = ("speed_kmh", "barrier", "angle_deg", "crush")
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")

metadata = MetaData()
incidents = Table(
    "incidents", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("received_at", DateTime(timezone=True), nullable=False),
    Column("device_id", String(64)),
    Column("event_type", String(24), nullable=False),
    Column("classification_source", String(8), nullable=False),
    Column("device_profile", String(32)),
    Column("device_severity_class", Integer),
    Column("server_profile", String(32)),
    Column("server_profile_thresholds", JSON),
    Column("server_profile_note", Text),
    Column("server_severity_class", Integer),
    Column("server_severity_name", String(16)),
    Column("p_crash", Float),
    Column("peak_g", Float),
    Column("excursion_ms", Float),
    Column("impulse_gs", Float),
    Column("signature_match", Boolean),
    Column("label_source", String(24)),
    Column("ground_truth", String(16)),
    Column("run_conditions", JSON),
    Column("device_metrics", JSON),
    Column("raw_window", JSON),
    Column("window_sha256", String(64)),
)

SUMMARY_COLS = [c for c in incidents.c if c.name != "raw_window"]


def performance_exclusion_reason(row):
    """None if the row may enter a model-performance statistic, else the reason it may not."""
    if row["event_type"] == "manual_panic":
        return "manual_panic"
    if row["server_profile"] != profiles.PRODUCTION or row["device_profile"] not in (None, profiles.PRODUCTION):
        return "non_production_profile"
    if row["event_type"] != "trigger":
        return row["event_type"]
    return None


def _engine():
    url = os.environ.get("EVENTS_DATABASE_URL", "sqlite:///" + str(Path(__file__).parent / "data" / "events.db"))
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(url, future=True)
    log.info("Incident store: %s", eng.url.render_as_string(hide_password=True))
    return eng


def _row_out(r, include_window=False):
    d = dict(r._mapping)
    d["received_at"] = d["received_at"].isoformat() if d.get("received_at") else None
    d["excluded_from_performance"] = performance_exclusion_reason(d)
    if not include_window:
        d.pop("raw_window", None)
    return d


def compute_stats(rows):
    """Confusion matrix and per-class / crash-detection metrics over eligible, labelled rows."""
    excluded, cm = {}, [[0] * 3 for _ in range(3)]
    for r in rows:
        reason = performance_exclusion_reason(r)
        if reason is None and r["ground_truth"] not in CLASS_NAMES:
            reason = "no_verified_label"
        if reason is None and r["server_severity_class"] is None:
            reason = "not_classified"
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        cm[CLASS_NAMES.index(r["ground_truth"])][r["server_severity_class"]] += 1
    n = sum(map(sum, cm))

    def ratio(a, b):
        return round(a / b, 4) if b else None

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        tp, fn, fp = cm[i][i], sum(cm[i]) - cm[i][i], sum(cm[j][i] for j in range(3)) - cm[i][i]
        p, rc = ratio(tp, tp + fp), ratio(tp, tp + fn)
        f1 = round(2 * p * rc / (p + rc), 4) if p and rc else (0.0 if p is not None and rc is not None else None)
        per_class[name] = {"precision": p, "recall": rc, "f1": f1, "support": sum(cm[i])}
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    crash_tp = sum(cm[i][j] for i in (1, 2) for j in (1, 2))
    crash_fn = sum(cm[i][0] for i in (1, 2))
    crash_fp = cm[0][1] + cm[0][2]
    cp, cr = ratio(crash_tp, crash_tp + crash_fp), ratio(crash_tp, crash_tp + crash_fn)
    return {
        "eligibility_rule": "event_type == 'trigger' AND server_profile == 'production' AND device_profile in "
                            "('production', null) AND ground_truth verified",
        "included_n": n, "excluded": excluded,
        "confusion_matrix": {"rows_true_cols_pred": CLASS_NAMES, "matrix": cm},
        "per_class": per_class,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if len(f1s) == 3 else None,
        "crash_detection": {"precision": cp, "recall": cr,
                            "f1": round(2 * cp * cr / (cp + cr), 4) if cp and cr else None},
    }


def create_blueprint(classify, parse_window, active_profile, engine=None):
    bp = Blueprint("events", __name__)
    eng = engine or _engine()
    metadata.create_all(eng)
    api_key = os.environ.get("EVENTS_API_KEY")
    if not api_key:
        log.warning("EVENTS_API_KEY not set: /api/v1/events writes are unauthenticated")

    def authorised():
        return not api_key or request.headers.get("X-API-Key") == api_key

    def respond(body, status=200):
        body.update(active_profile().response_fields())
        resp = jsonify(body)
        resp.headers["X-Threshold-Profile"] = active_profile().name
        return resp, status

    def err(msg, status=400):
        return respond({"error": msg}, status)

    @bp.route("/api/v1/events", methods=["POST"])
    def create_event():
        if not authorised():
            return err("invalid or missing X-API-Key", 401)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return err("body must be a JSON object")
        etype = data.get("event_type", "trigger")
        if etype not in EVENT_TYPES:
            return err(f"event_type must be one of {EVENT_TYPES}")
        source = data.get("classification_source", "none")
        if source not in CLASSIFICATION_SOURCES:
            return err(f"classification_source must be one of {CLASSIFICATION_SOURCES}")
        dev_cls = data.get("device_severity_class")
        if dev_cls is not None and dev_cls not in (0, 1, 2):
            return err("device_severity_class must be 0, 1 or 2")
        truth, run = data.get("ground_truth"), data.get("run_conditions")
        if run is not None and not isinstance(run, dict):
            return err("run_conditions must be an object")
        metrics = data.get("device_metrics")
        if metrics is not None and not isinstance(metrics, dict):
            return err("device_metrics must be an object")
        if etype == "rc_collection":
            # unlabelled runs are kept (label later with PATCH); analysis ignores them until labelled
            if truth is not None and truth not in RC_LABELS:
                return err(f"rc_collection ground_truth must be one of {RC_LABELS}")
            missing = [k for k in RC_RUN_KEYS if not run or run.get(k) in (None, "")]
            if missing:
                return err(f"rc_collection requires run_conditions {missing}")
        elif truth is not None and truth not in CLASS_NAMES:
            return err(f"ground_truth must be one of {CLASS_NAMES}")

        has_window = any(k in data for k in CHANNELS)
        if not has_window and etype != "manual_panic":
            return err("raw window (ax, ay, az, gx, gy, gz; 500 samples each) is required")
        row = dict(received_at=datetime.now(timezone.utc), device_id=str(data.get("device_id") or "")[:64] or None,
                   event_type=etype, classification_source=source,
                   device_profile=(str(data["device_profile"]).lower()[:32] if data.get("device_profile") else None),
                   device_severity_class=dev_cls, ground_truth=truth, run_conditions=run,
                   device_metrics=metrics)
        prof = active_profile()
        row.update(server_profile=prof.name, server_profile_thresholds=prof.thresholds(),
                   server_profile_note=prof.note)
        result = None
        if has_window:
            arrays, problem = parse_window(data)
            if problem:
                return err(problem)
            window = {k: arrays[k].tolist() for k in CHANNELS}
            canon = json.dumps(window, separators=(",", ":"), sort_keys=True)
            result = classify(arrays)
            sig = result["crash_signature"]
            row.update(raw_window=window, window_sha256=hashlib.sha256(canon.encode()).hexdigest(),
                       server_severity_class=result["severity_class"], server_severity_name=result["severity_name"],
                       p_crash=result["p_crash"], peak_g=sig["peak_g"], excursion_ms=sig["excursion_ms"],
                       impulse_gs=sig["impulse_gs"], signature_match=sig["signature_match"],
                       label_source=result["label_source"])
        with eng.begin() as c:
            new_id = c.execute(incidents.insert().values(**row)).inserted_primary_key[0]
        return respond({"id": new_id, "event_type": etype, "device_profile": row["device_profile"],
                        "server_profile": prof.name, "raw_window_stored": has_window,
                        "excluded_from_performance": performance_exclusion_reason(row),
                        "classification": result}, 201)

    @bp.route("/api/v1/events", methods=["GET"])
    def list_events():
        q = select(*SUMMARY_COLS).order_by(incidents.c.id.desc()).limit(min(int(request.args.get("limit", 100)), 1000))
        for f in ("event_type", "server_profile", "device_profile"):
            if request.args.get(f):
                q = q.where(incidents.c[f] == request.args[f])
        with eng.connect() as c:
            rows = [_row_out(r) for r in c.execute(q)]
        return respond({"events": rows, "count": len(rows)})

    @bp.route("/api/v1/events/<int:event_id>", methods=["GET"])
    def get_event(event_id):
        with eng.connect() as c:
            r = c.execute(select(incidents).where(incidents.c.id == event_id)).first()
        return respond({"event": _row_out(r, include_window=True)}) if r else err("not found", 404)

    @bp.route("/api/v1/events/<int:event_id>", methods=["PATCH"])
    def label_event(event_id):
        if not authorised():
            return err("invalid or missing X-API-Key", 401)
        truth = (request.get_json(silent=True) or {}).get("ground_truth")
        with eng.begin() as c:
            r = c.execute(select(incidents.c.event_type).where(incidents.c.id == event_id)).first()
            if not r:
                return err("not found", 404)
            allowed = RC_LABELS if r.event_type == "rc_collection" else CLASS_NAMES
            if truth not in allowed:
                return err(f"ground_truth must be one of {allowed}")
            c.execute(incidents.update().where(incidents.c.id == event_id).values(ground_truth=truth))
        return respond({"id": event_id, "ground_truth": truth})

    @bp.route("/api/v1/events/export", methods=["GET"])
    def export_events():
        """Full rows including raw windows, one JSON object per line (input to analysis/)."""
        q = select(incidents).order_by(incidents.c.id)
        if request.args.get("event_type"):
            q = q.where(incidents.c.event_type == request.args["event_type"])
        with eng.connect() as c:
            lines = [json.dumps(_row_out(r, include_window=True)) for r in c.execute(q)]
        resp = Response("\n".join(lines) + ("\n" if lines else ""), mimetype="application/x-ndjson")
        resp.headers["X-Threshold-Profile"] = active_profile().name
        return resp

    @bp.route("/api/v1/stats", methods=["GET"])
    def stats():
        cols = [incidents.c[k] for k in ("event_type", "server_profile", "device_profile",
                                         "ground_truth", "server_severity_class")]
        with eng.connect() as c:
            rows = [dict(r._mapping) for r in c.execute(select(*cols))]
        return respond({"performance": compute_stats(rows)})

    return bp
