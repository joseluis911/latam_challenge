# syntax=docker/dockerfile:1.6
# ----------------------------------------------------------------------------
# Multi-stage build for the LATAM Flight Delay Prediction API.
# Target: Google Cloud Run (listens on $PORT, defaults to 8080).
# ----------------------------------------------------------------------------

# ---- Stage 1: build wheels ----
FROM python:3.10-slim AS builder

WORKDIR /build

# System deps needed only at build time (compile numpy/scipy/pandas wheels if missing).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt


# ---- Stage 2: runtime ----
FROM python:3.10-slim

# Non-root user for the runtime container.
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Install runtime deps from prebuilt wheels (no compilers in final image).
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code and bundled training data.
COPY --chown=app:app challenge ./challenge
COPY --chown=app:app data ./data

ENV PORT=8080 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

USER app

# Cloud Run injects $PORT; uvicorn binds to it on 0.0.0.0.
CMD exec uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT}
