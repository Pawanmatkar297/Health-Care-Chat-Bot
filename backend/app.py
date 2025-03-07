from flask import Flask, request, jsonify
from flask_cors import CORS
from .chatbot import HealthcareChatbot
import traceback
import os
from dotenv import load_dotenv

import nltk
print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')
print("NLTK data downloaded successfully")

app = Flask(__name__)
load_dotenv()  # Load environment variables from .env file
CORS(app, resources={r"/*": {"origins": "*"}})

print("Initializing chatbot...")
# Initialize chatbot
chatbot = HealthcareChatbot()
print("Chatbot initialized successfully")

# Store symptoms for each session
symptoms_dict = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    print("Headers:", request.headers)
    print("Body:", request.get_json())
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
            # Process symptoms and generate diagnosis
            matching_diseases = chatbot.find_matching_diseases(symptoms_dict[session_id])
            if matching_diseases.empty:
                response = ("I couldn't find any matching conditions for your symptoms. Please consult a healthcare professional." 
                          if language == 'en' 
                          else "मैं आपके लक्षणों से मेल खाती कोई स्थिति नहीं ढूंढ पाया। कृपया चिकित्सक से परामर्श करें।")
            else:
                disease_match_count = matching_diseases[chatbot.symptom_columns].apply(
                    lambda x: x.str.contains('|'.join([
                        s['processed'] if isinstance(s, dict) else s 
                        for s in symptoms_dict[session_id]
                    ]), case=False, na=False).sum(), 
                    axis=1
                )
                best_match = matching_diseases.loc[disease_match_count.idxmax()]
                response = chatbot.generate_response(best_match, language)
            
            # Clear symptoms list for next conversation
            symptoms_dict[session_id] = []
            
            return jsonify({
                'success': True,
                'message': response,
                'is_final': True,
                'language': language
            })

        # Process the symptom with language support
        preprocessed_input = chatbot.preprocess_text(message, language)
        symptoms_dict[session_id].append(preprocessed_input)
        
        continue_message = ("Got it. Any other symptoms? Say 'no' if you're finished." 
                          if language == 'en' 
                          else "समझ गया। कोई अन्य लक्षण? यदि आप समाप्त कर चुके हैं तो 'नहीं' कहें।")
        
        return jsonify({
            'success': True,
            'message': continue_message,
            'is_final': False,
            'language': language
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5002)))