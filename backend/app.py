from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import HealthcareChatbot
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5000"]}})

# Initialize chatbot
chatbot = HealthcareChatbot()

# Store symptoms for each session
symptoms_dict = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').lower()
        session_id = data.get('session_id', 'default')

        # Initialize symptoms list for new session
        if session_id not in symptoms_dict:
            symptoms_dict[session_id] = []

        # Check if user wants to end symptom collection
        if message.lower() in ['no', 'nope', 'done']:
            if not symptoms_dict[session_id]:
                return jsonify({
                    'success': True,
                    'message': "No symptoms were provided. Please tell me your symptoms.",
                    'is_final': False
                })
            
            # Process symptoms and generate diagnosis
            matching_diseases = chatbot.find_matching_diseases(symptoms_dict[session_id])
            if matching_diseases.empty:
                response = "I couldn't find any matching conditions for your symptoms. Please consult a healthcare professional."
            else:
                disease_match_count = matching_diseases[chatbot.symptom_columns].apply(
                    lambda x: x.str.contains('|'.join(symptoms_dict[session_id]), case=False, na=False).sum(), 
                    axis=1
                )
                best_match = matching_diseases.loc[disease_match_count.idxmax()]
                response = chatbot.generate_response(best_match)
            
            # Clear symptoms list for next conversation
            symptoms_dict[session_id] = []
            
            return jsonify({
                'success': True,
                'message': response,
                'is_final': True
            })

        # Process the symptom
        preprocessed_input = chatbot.preprocess_text(message)
        symptoms_dict[session_id].append(preprocessed_input)
        
        return jsonify({
            'success': True,
            'message': f"Got it. Any other symptoms? Say 'no' if you're finished.",
            'is_final': False
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': "Sorry, I encountered an error. Please try again."
        })

if __name__ == '__main__':
    app.run(debug=True, port=5002) 