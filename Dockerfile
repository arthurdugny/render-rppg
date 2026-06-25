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

COPY server.py model.py resnet.py ./
COPY 79999_iter.pth ./

# rppg_live.html optionnel — seulement si vous servez le HTML depuis ce serveur
# (sinon Lovable héberge déjà le HTML, et ce backend ne fait que le WS + l'API)
COPY rppg_live.html ./

# Render fournit la variable PORT automatiquement
ENV PORT=8000
EXPOSE 8000

CMD ["python", "server.py"]
