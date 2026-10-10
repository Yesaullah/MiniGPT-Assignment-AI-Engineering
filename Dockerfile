# CPU-only inference image for Render (free tier: 0.1 CPU, 512 MB RAM).
FROM python:3.12.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_THREADS=1
WORKDIR /app

RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir fastapi==0.136.1 uvicorn==0.46.0 pydantic==2.13.3 pyyaml==6.0.3

COPY src ./src
COPY configs ./configs
COPY checkpoints/RELEASE.json ./checkpoints/RELEASE.json
# Download the released model at build time and verify its SHA-256 (fails the build on mismatch).
RUN python -m src.artifacts fetch

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.serve.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
