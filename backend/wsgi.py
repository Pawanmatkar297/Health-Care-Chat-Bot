import os
from app import app

# This is the line that Gunicorn uses
application = app

# Render will provide the PORT
app.config['PORT'] = 10000

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False) 