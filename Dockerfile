FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLAME_DETECT_DATA_DIR=/app/data \
    FLAME_DETECT_PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY main.py index.html ./

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    numpy \
    opencv-python-headless \
    ultralytics \
    torch \
    psutil

RUN mkdir -p /app/data/history /app/data/logs /app/data/models /app/data/reports /app/data/uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]