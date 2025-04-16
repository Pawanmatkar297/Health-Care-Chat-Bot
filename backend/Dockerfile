FROM python:3.11-slim

WORKDIR /app

# Download NLTK data during build
RUN python -m pip install --upgrade pip && \
    pip install nltk && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Copy only requirements first
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend directory contents
COPY backend/ .

# Ensure the JSON file is in the right location
RUN ls -la

# Default port
ENV PORT=10000

# Expose the port
EXPOSE ${PORT}

# Start command with a single worker to reduce memory usage
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 app:app 