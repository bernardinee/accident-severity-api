FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and application code
# artifacts/ holds the Phase 3 v2 model the app loads; models/ is the previous
# v1 model, kept unused for one release so a rollback needs no image rebuild.
COPY models/ models/
COPY artifacts/ artifacts/
COPY app.py .

EXPOSE $PORT

# Railway injects $PORT; fall back to 5000 for local Docker runs
CMD gunicorn --workers 2 --bind "0.0.0.0:${PORT:-5000}" --timeout 30 --access-logfile - app:app
