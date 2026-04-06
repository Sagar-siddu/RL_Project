# Dynamic Pricing Environment — Dockerfile
# Builds on openenv-base for compatibility with the OpenEnv ecosystem.
# Works as a standalone image too (falls back to pip install openenv-core).

ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install Python dependencies
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy environment source code
COPY models.py           /app/models.py
COPY market_simulator.py /app/market_simulator.py
COPY tasks.py            /app/tasks.py
COPY server/             /app/server/

WORKDIR /app

# Health check — OpenEnv validator pings /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# HuggingFace Spaces uses port 7860
EXPOSE 7860

ENV PYTHONPATH=/app
ENV PRICING_TASK=single_sku_stable

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
