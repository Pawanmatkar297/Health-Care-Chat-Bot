import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

# This is the object that gunicorn looks for
application = app

# Get port from environment variable or default to 10000
port = int(os.environ.get("PORT", 10000))

if __name__ == "__main__":
    # Bind to PORT if defined, otherwise default to 10000
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    ) 