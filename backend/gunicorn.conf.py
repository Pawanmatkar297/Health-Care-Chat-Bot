import os

# Get port from environment variable or default to 10000
port = int(os.environ.get("PORT", 10000))

# Bind to 0.0.0.0 to allow external access
bind = f"0.0.0.0:{port}"

# Worker configuration
workers = 4
worker_class = "sync"
threads = 2

# Timeout configuration
timeout = 120

# SSL configuration (if needed)
keyfile = None
certfile = None

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info" 