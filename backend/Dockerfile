FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 10000 available to the world outside this container
EXPOSE 10000

# Set environment variable for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=10000

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app 