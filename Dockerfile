# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps (needed for some eth/crypto packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libssl-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY env_helpers.py       .
COPY polymarket_client.py .
COPY claude_analyst.py    .
COPY strategy.py          .
COPY risk_manager.py      .
COPY main.py              .

# ── Security: run as non-root user ────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash botuser \
    # Give botuser ownership of the app dir (needed to write .pyc cache etc.)
    && chown -R botuser:botuser /app \
    # Dedicated writable log dir, separate from the app source
    && mkdir -p /var/log/polybot \
    && chown botuser:botuser /var/log/polybot

USER botuser

# ── Environment ───────────────────────────────────────────────────────────────
# These are defaults only — override at runtime via --env-file or -e flags.
# Never bake real secrets into the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POLYMARKET_SIGNATURE_TYPE=1 \
    LOG_LEVEL=INFO \
    LOG_FILE=/var/log/polybot/bot.log

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Verifies the app can at least import its modules cleanly.
HEALTHCHECK --interval=60s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from polymarket_client import PolymarketClient; print('ok')"

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "main.py"]