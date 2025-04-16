FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install NLTK and download all required data
RUN pip install --no-cache-dir nltk && \
    mkdir -p /usr/share/nltk_data/corpora && \
    mkdir -p /usr/share/nltk_data/tokenizers && \
    mkdir -p /usr/share/nltk_data/sentiment && \
    python -c "import nltk; nltk.download('punkt', download_dir='/usr/share/nltk_data'); nltk.download('stopwords', download_dir='/usr/share/nltk_data'); nltk.download('vader_lexicon', download_dir='/usr/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/share/nltk_data')"

# Set environment variable for NLTK data path
ENV NLTK_DATA=/usr/share/nltk_data

# Copy requirements and install dependencies
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a simplified wrapper script that will handle initialization errors gracefully
COPY backend/app.py .
COPY backend/chatbot ./chatbot
COPY backend/ml_model.py .
COPY backend/disease_prevalence.json .
COPY backend/wsgi.py .
COPY backend/gunicorn.conf.py .
COPY backend/Priority_wise_MedicalDataset* ./

# Create models directory
RUN mkdir -p models

# Default port
ENV PORT=10000
EXPOSE ${PORT}

# Create a wrapper script
RUN echo '#!/usr/bin/env python3\ntry:\n    from app import app\nexcept Exception as e:\n    import sys\n    from flask import Flask\n    print(f"Error loading app: {e}", file=sys.stderr)\n    app = Flask(__name__)\n    @app.route("/")\n    def home():\n        return "Health Chat Bot API (Error fallback mode)"\n    @app.route("/ping")\n    def ping():\n        return "pong"\n' > wrapper.py

# Start command with wrapper
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 wrapper:app 