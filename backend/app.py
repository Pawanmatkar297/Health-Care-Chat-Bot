from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chatbot.chatbot import MedicalChatbot
import traceback
from dotenv import load_dotenv

import nltk
print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')  # Required for sentiment analysis
print("NLTK data downloaded successfully")

app = Flask(__name__)
load_dotenv()  # Load environment variables from .env file

# Configure CORS with all necessary settings
CORS(app, 
     resources={
         r"/*": {
             "origins": [
                 "http://localhost:3000"
             ],
             "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
             "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"],
             "expose_headers": ["Content-Type", "Authorization"],
             "supports_credentials": True,
             "max_age": 120
         }
     })

print("Initializing chatbot...")
# Initialize chatbot with English as default language
chatbot = MedicalChatbot(language='english')
print("Chatbot initialized successfully")

# Store symptoms for each session
symptoms_dict = {}

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    allowed_origins = ["http://localhost:3000"]
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
    allowed_origins = ["http://localhost:3000"]
    if origin in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Authorization')
    return response

@app.route('/api/chat', methods=['POST'])
def chat():
    print("Headers:", request.headers)
    print("Body:", request.get_json())
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        language = data.get('language', 'en')  # Get language preference
        
        # Set chatbot language based on user preference
        chatbot.set_language('hindi' if language == 'hi' else 'english')
        
        print(f"Received message: {message}")
        print(f"Session ID: {session_id}")
        print(f"Language: {language}")

        # Initialize symptoms list for new session
        if session_id not in symptoms_dict:
            symptoms_dict[session_id] = []

        # Process the user input based on language
        processed_input = chatbot.process_user_input(message)

        # Check if user wants to end symptom collection
        if processed_input.lower() in ['no', 'nope', 'done', 'नहीं', 'बस', 'नही', 'no more']:
            if not symptoms_dict[session_id]:
                return jsonify({
                    'success': True,
                    'message': chatbot.get_phrase('at_least_one'),
                    'is_final': False
                })
            
            print(f"Processing symptoms: {symptoms_dict[session_id]}")
            # Flatten nested lists and ensure all elements are strings
            flat_symptoms = []
            for symptom in symptoms_dict[session_id]:
                if isinstance(symptom, list):
                    flat_symptoms.extend([s for s in symptom if isinstance(s, str)])
                elif isinstance(symptom, str):
                    flat_symptoms.append(symptom)
            
            print(f"Flattened symptoms: {flat_symptoms}")
            if not flat_symptoms:
                return jsonify({
                    'success': True,
                    'message': chatbot.get_phrase('valid_symptom'),
                    'is_final': False
                })
                
            results = chatbot.process_symptoms(flat_symptoms)
            response = chatbot.generate_response(results)
            
            # Clear symptoms list for next conversation
            symptoms_dict[session_id] = []
            
            return jsonify({
                'success': True,
                'message': response,
                'is_final': True,
                'language': language
            })

        # Process and validate the symptom
        preprocessed_input = chatbot.preprocess_text(processed_input)
        if preprocessed_input:
            # Handle both string and dictionary responses from preprocess_text
            if isinstance(preprocessed_input, dict):
                symptoms_dict[session_id].append(preprocessed_input['processed'])
            else:
                symptoms_dict[session_id].append(preprocessed_input)
            continue_message = chatbot.get_phrase('enter_another')
        else:
            continue_message = chatbot.get_phrase('valid_symptom')

        return jsonify({
            'success': True,
            'message': continue_message,
            'is_final': False,
            'language': language
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        error_message = chatbot.translate_text(
            "Sorry, I encountered an error. Please try again.",
            'hindi' if language == 'hi' else 'english'
        )
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5002)))