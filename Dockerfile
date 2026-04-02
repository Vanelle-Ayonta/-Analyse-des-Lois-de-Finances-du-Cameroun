# Dockerfile
FROM python:3.11-slim

# Dépendances système (PyMuPDF, Pillow, wordcloud, torch)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer Poetry
RUN pip install --no-cache-dir poetry==1.8.3
RUN poetry config virtualenvs.create false

# Dépendances en premier (optimise le cache Docker)
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only=main --no-root --no-interaction --no-ansi

# Code et données
COPY . .

# Dossier modèle vide (sera rempli au premier lancement HuggingFace)
RUN mkdir -p modele

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.headless=true", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
