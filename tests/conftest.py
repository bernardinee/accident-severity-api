import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# app.py opens the incident store at import; keep test runs out of data/events.db
os.environ["EVENTS_DATABASE_URL"] = "sqlite:///" + str(Path(tempfile.mkdtemp()) / "events_test.db").replace("\\", "/")
os.environ.pop("THRESHOLD_PROFILE", None)
os.environ.pop("SIG_TRANSIENT_MIN_MS", None)
os.environ.pop("EVENTS_API_KEY", None)
