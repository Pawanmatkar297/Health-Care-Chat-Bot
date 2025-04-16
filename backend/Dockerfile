FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend directory contents
COPY backend/ .

# Default port
ENV PORT=10000

# Expose the port
EXPOSE ${PORT}

# Start command
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 4 --threads 2 app:app 