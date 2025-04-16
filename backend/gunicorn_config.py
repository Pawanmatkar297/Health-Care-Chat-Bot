import os

# Bind to 0.0.0.0 to make the server publicly accessible
port = os.environ.get("PORT", 10000)
bind = f"0.0.0.0:{port}"

# Worker configuration
workers = 4
threads = 2

# Timeout configuration
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# SSL configuration (if needed)
# keyfile = 'path/to/keyfile'
# certfile = 'path/to/certfile'

# Prevent timeout issues
worker_class = "sync"
worker_connections = 1000
timeout = 300
graceful_timeout = 300 