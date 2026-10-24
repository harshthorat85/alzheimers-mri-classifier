FROM python:3.10-slim

# HF Spaces runs containers as a non-root user (UID 1000).
# Creating the user explicitly prevents permission errors on mounted volumes.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/user/app

# Copy and install dependencies first (separate layer = faster rebuilds on code changes)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# HF Spaces requires port 7860 — correct
EXPOSE 7860

# Use 2 workers for slight throughput improvement on CPU inference.
# --timeout 120 prevents uvicorn killing slow model-load cold starts.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2", "--timeout-keep-alive", "120"]