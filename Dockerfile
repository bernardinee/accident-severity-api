FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and application code
# artifacts/decision_config.json names the model the app loads (Phase 5:
# p5_xgboost.joblib + calibration + thresholds). artifacts/phase2_* is the
# previous (Phase 3 v2) model and models/ the v1 model, both kept unused so a
# rollback is a revert of app.py/decision_config.json with no artifact upload.
COPY models/ models/
COPY artifacts/ artifacts/
COPY app.py profiles.py events.py .

# Incident store defaults to sqlite:///data/events.db. The container filesystem is
# wiped on redeploy: mount a Railway volume at /app/data (or set
# EVENTS_DATABASE_URL) before collecting anything that must survive.

EXPOSE $PORT

# Railway injects $PORT; fall back to 5000 for local Docker runs
CMD gunicorn --workers 2 --bind "0.0.0.0:${PORT:-5000}" --timeout 30 --access-logfile - app:app
