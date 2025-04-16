import os

# Basic configuration
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
workers = 4
threads = 2
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Reload workers when code changes (development only)
reload = False

# Process naming
proc_name = 'healthcare-chatbot'

# Maximum number of requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50

# SSL configuration (if needed)
# keyfile = 'path/to/keyfile'
# certfile = 'path/to/certfile' 