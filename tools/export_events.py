"""Download stored incidents (with raw windows) as JSON Lines.

    python tools/export_events.py --event-type rc_collection            # -> data/rc_collection/events.jsonl
    python tools/export_events.py --url http://localhost:5000 --out x.jsonl
"""
import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROD_URL = "https://accident-severity-api-production.up.railway.app"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=PROD_URL)
    ap.add_argument("--event-type", default="rc_collection")
    ap.add_argument("--out", default=str(ROOT / "data" / "rc_collection" / "events.jsonl"))
    a = ap.parse_args()
    q = f"?event_type={a.event_type}" if a.event_type else ""
    with urllib.request.urlopen(a.url.rstrip("/") + "/api/v1/events/export" + q, timeout=120) as r:
        body = r.read().decode()
        profile = r.headers.get("X-Threshold-Profile")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8", newline="\n")
    print(f"{len(body.splitlines())} events -> {out} (server profile at export: {profile})")


if __name__ == "__main__":
    main()
