from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)

# Configure CORS with all necessary settings
CORS(app, 
     resources={
         r"/*": {
             "origins": ["*"],  # Allow all origins in production
             "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
             "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"],
             "expose_headers": ["Content-Type", "Authorization"],
             "supports_credentials": True,
             "max_age": 120
         }
     })

# Get port from environment variable
port = int(os.environ.get("PORT", 10000))
print(f"Configured to use port: {port}")

# Add a ping endpoint that responds immediately
@app.route('/ping')
def ping():
    return 'pong'

# Add a health check endpoint
@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "message": "Service is running"}), 200

# Simple endpoint for chat
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        language = data.get('language', 'en')  # Get language preference
        
        print(f"Received message: {message}")
        print(f"Language: {language}")

        # Provide a simple response based on keywords
        response = "I'm a simplified version of the healthcare chatbot. The ML model is being initialized. Please try again later or contact support."
        
        return jsonify({
            'success': True,
            'message': response,
            'is_final': True,
            'language': language
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"An error occurred: {str(e)}",
            'language': 'en'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port) 