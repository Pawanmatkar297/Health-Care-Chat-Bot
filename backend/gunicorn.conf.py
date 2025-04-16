import os

# Get port from environment variable or default to 10000
port = int(os.environ.get("PORT", 10000))

# Bind to 0.0.0.0 to allow external access
bind = f"0.0.0.0:{port}"

# Worker configuration
workers = 1  # Reduced number of workers for ML application
worker_class = "sync"
threads = 2

# Timeout configuration
timeout = 300  # Increased timeout for ML operations
graceful_timeout = 300

# SSL configuration (if needed)
keyfile = None
certfile = None

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Prevent timeout issues
keepalive = 65
worker_connections = 2000

# Startup configuration
preload_app = True
reload = False 