"""Recover events the device could not upload.

The firmware prints the complete event as "[UNSENT_EVENT] {json}" whenever the
POST to /api/v1/events fails (no WiFi, server down). This re-posts those lines
from a captured serial log, so no window is lost.

    python tools/ingest_serial_log.py session.log                      # POST to production
    python tools/ingest_serial_log.py session.log --url http://localhost:5000
    python tools/ingest_serial_log.py session.log --dry-run            # just count and validate
Set EVENTS_API_KEY in the environment if the server requires it.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

PROD_URL = "https://accident-severity-api-production.up.railway.app"
MARK = "[UNSENT_EVENT] "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--url", default=PROD_URL)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    events = []
    with open(a.log, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            i = line.find(MARK)
            if i < 0:
                continue
            try:
                events.append((n, json.loads(line[i + len(MARK):])))
            except json.JSONDecodeError as e:
                print(f"line {n}: truncated or corrupt event ({e}); skipped")
    print(f"{len(events)} unsent events in {a.log}")
    if a.dry_run:
        return
    headers = {"Content-Type": "application/json"}
    if os.environ.get("EVENTS_API_KEY"):
        headers["X-API-Key"] = os.environ["EVENTS_API_KEY"]
    failed = 0
    for n, ev in events:
        req = urllib.request.Request(a.url.rstrip("/") + "/api/v1/events", json.dumps(ev).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read())
                print(f"line {n}: stored id={body['id']} server_profile={body['server_profile']}")
        except urllib.error.HTTPError as e:
            failed += 1
            print(f"line {n}: HTTP {e.code} {e.read().decode()[:200]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
