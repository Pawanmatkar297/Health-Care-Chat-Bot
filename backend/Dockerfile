FROM python:3.11-slim

# Install system dependencies including espeak
RUN apt-get update && apt-get install -y \
    espeak \
    espeak-ng \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=10000

# Expose the port
EXPOSE $PORT

# Command to run the application
CMD ["python", "run.py"] 