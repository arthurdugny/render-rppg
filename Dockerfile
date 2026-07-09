# Dockerfile — serveur rPPG (FastAPI + BiSeNet)
FROM python:3.10-slim

# Dépendances système pour opencv-python-headless
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py cnn1d_model.py ./
COPY inference_paume_video.py extract_rgb_paume_multi.py metrics.py ./
COPY best_cnn1d.pth ./
COPY rppg_live.html ./

# Render fournit la variable PORT automatiquement
ENV PORT=8000
EXPOSE 8000

CMD ["python", "server.py"]
