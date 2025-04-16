import os
import multiprocessing

# Bind to 0.0.0.0 to make the server publicly accessible
bind = f"0.0.0.0:{int(os.environ.get('PORT', 10000))}"

# Worker configuration
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
threads = 4

# Timeout configuration
timeout = 120
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

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