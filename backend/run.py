import os
from app import app

# Get port from environment variable
port = int(os.environ.get("PORT", 10000))
print(f"Starting server on port {port}")

if __name__ == "__main__":
    # Bind to PORT if defined, otherwise default to 10000
    app.run(host='0.0.0.0', port=port) 