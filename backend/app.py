from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback
from dotenv import load_dotenv

# Create Flask app first
app = Flask(__name__)

# Add a basic route that will always work, even if other initialization fails
@app.route('/')
def root():
    return jsonify({
        'status': 'online',
        'message': 'MediAssist API is running'
    })

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try/except for the rest of the imports and initialization
try:
    from chatbot import MedicalChatbot

    import nltk
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')  # Add wordnet for lemmatization
    nltk.download('vader_lexicon')  # Add VADER lexicon for sentiment analysis
    print("NLTK data downloaded successfully")

    load_dotenv()  # Load environment variables from .env file

    # Configure CORS with all necessary settings
    CORS(app, 
         resources={
             r"/*": {
                 "origins": [
                     "https://mediassist-o69d.onrender.com",
                     "http://localhost:3000",
                     "https://mediassist-o69d.onrender.com"
                 ],
                 "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
                 "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"],
                 "expose_headers": ["Content-Type", "Authorization"],
                 "supports_credentials": True,
                 "max_age": 120
             }
         })

    # Add a test endpoint
    @app.route('/api/test', methods=['GET'])
    def test():
        return jsonify({
            'status': 'success',
            'message': 'Backend is running and accessible'
        })

    print("Initializing chatbot...")
    # Initialize chatbot
    chatbot = MedicalChatbot()
    print("Chatbot initialized successfully")

    # Store symptoms for each session
    symptoms_dict = {}

    # Add CORS headers to all responses
    @app.after_request
    def after_request(response):
        origin = request.headers.get('Origin')
        allowed_origins = [
            "https://mediassist-o69d.onrender.com",
            "http://localhost:3000",
            "https://mediassist-o69d.onrender.com"
        ]
        if origin in allowed_origins:
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Authorization')
        return response

    # Handle OPTIONS requests
    @app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
    @app.route('/<path:path>', methods=['OPTIONS'])
    def handle_options(path):
        response = jsonify({"status": "ok"})
        origin = request.headers.get('Origin')
        allowed_origins = [
            "https://mediassist-o69d.onrender.com",
            "http://localhost:3000",
            "https://mediassist-o69d.onrender.com"
        ]
        if origin in allowed_origins:
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Authorization')
        return response

    @app.route('/api/chat', methods=['POST'])
    def chat():
        print("=== New Chat Request ===")
        print("Headers:", request.headers)
        print("Body:", request.get_json())
        print("Origin:", request.headers.get('Origin'))
        print("Content-Type:", request.headers.get('Content-Type'))
        print("======================")
        try:
            data = request.get_json()
            message = data.get('message', '').strip()
            session_id = data.get('session_id', 'default')
            language = data.get('language', 'en')  # Get language preference
            
            print(f"Received message: {message}")
            print(f"Session ID: {session_id}")
            print(f"Language: {language}")

            # Initialize symptoms list for new session
            if session_id not in symptoms_dict:
                symptoms_dict[session_id] = []

            # Check if user wants to end symptom collection
            if message.lower() in ['no', 'nope', 'done', 'नहीं', 'बस', 'नही']:
                if not symptoms_dict[session_id]:
                    return jsonify({
                        'success': True,
                        'message': "No symptoms were provided. Please tell me your symptoms." if language == 'en' else "कोई लक्षण नहीं बताया गया। कृपया अपने लक्षण बताएं।",
                        'is_final': False
                    })
                
                print(f"Processing symptoms: {symptoms_dict[session_id]}")
                # Process symptoms and generate diagnosis using MedicalChatbot methods
                results = chatbot.process_symptoms(symptoms_dict[session_id])
                response = chatbot.generate_response(results)
                
                # Clear symptoms list for next conversation
                symptoms_dict[session_id] = []
                
                return jsonify({
                    'success': True,
                    'message': response,
                    'is_final': True,
                    'final_output': response,  # Add final_output for frontend
                    'language': language
                })

            # Process the symptom
            corrected_input = chatbot.correct_spelling(message)
            symptoms_dict[session_id].append(corrected_input)
            
            continue_message = ("Got it. Any other symptoms? Say 'no' if you're finished." 
                              if language == 'en' 
                              else "समझ गया। कोई अन्य लक्षण? यदि आप समाप्त कर चुके हैं तो 'नहीं' कहें।")
            
            return jsonify({
                'success': True,
                'message': continue_message,
                'is_final': False,
                'language': language,
                'recognized_text': corrected_input  # Add recognized_text for frontend
            })

        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            error_message = ("Sorry, I encountered an error. Please try again." 
                            if language == 'en' 
                            else "क्षमा करें, एक त्रुटि हुई। कृपया पुनः प्रयास करें।")
            return jsonify({
                'success': False,
                'message': error_message,
                'language': language
            })
        

    @app.route('/api/chat-history/save', methods=['POST'])
    def save_chat_history():
        try:
            data = request.get_json()
            # Process the data and save it to your database or file
            return jsonify({'success': True, 'message': 'Chat history saved successfully'})
        except Exception as e:
            print(f"Error saving chat history: {str(e)}")
            return jsonify({'success': False, 'message': 'Failed to save chat history'})

except Exception as e:
    print(f"Error during initialization: {e}")
    print(traceback.format_exc())
    
    # Add fallback routes if initialization fails
    @app.route('/api/test', methods=['GET'])
    def test_fallback():
        return jsonify({
            'status': 'limited',
            'message': 'Backend is running with limited functionality due to initialization errors',
            'error': str(e)
        })

# This block only runs when directly executing the file with Python
if __name__ == '__main__':
    # For local development or direct Python execution
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting development server on port {port}")
    print(f"Server will be accessible at http://0.0.0.0:{port}")
    
    app.run(
        host='0.0.0.0',  # Bind to all available network interfaces
        port=port,
        debug=False      # Disable debug mode in production
    )
else:
    # When imported by Gunicorn or other WSGI server
    # Just print information about the environment
    port = os.environ.get("PORT", "Not Set")
    print(f"Flask application loaded by a WSGI server (Gunicorn)")
    print(f"Environment PORT variable: {port}")
    print(f"NOTE: When using gunicorn -b 0.0.0.0:$PORT, make sure PORT is set in environment")
    print(f"Ready to serve requests")

# Print a message to indicate the app has been loaded
print(f"Flask application initialization complete")