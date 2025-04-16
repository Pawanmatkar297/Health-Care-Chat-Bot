import os

# Bind to 0.0.0.0 to make the server publicly accessible
bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"

# Worker configuration
workers = 4
worker_class = 'sync'
threads = 2

# Timeout configuration
timeout = 120
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Reload workers when code changes (development only)
reload = False

# SSL configuration (if needed)
# keyfile = 'path/to/keyfile'
# certfile = 'path/to/certfile' 