FROM python:3.11-slim

WORKDIR /app

# Install only the minimal requirements
RUN pip install --no-cache-dir flask flask-cors gunicorn

# Copy only what's needed for the simplified app
COPY backend/simplified_app.py .

# Default port
ENV PORT=10000

# Expose the port
EXPOSE ${PORT}

# Start command with the simplified app
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 30 simplified_app:app 