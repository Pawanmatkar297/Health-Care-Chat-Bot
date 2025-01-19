import pyttsx3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import os

class HealthcareChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Use absolute path for the dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, 'MedDatasetFinal_modeled.csv')
        self.intents_df = self.load_intents(dataset_path)
        self.symptom_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']

    def load_intents(self, file_path):
        df = pd.read_csv(file_path)
        print("Columns in the CSV file:", df.columns)
        print("First few rows of the dataframe:")
        print(df.head())
        return df

    def text_to_speech(self, text):
        try:
            if not hasattr(self, 'tts_engine'):
                self.tts_engine = pyttsx3.init()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error in text_to_speech: {str(e)}")
            # Don't raise the error, just log it
            pass

    def preprocess_text(self, text):
        try:
            # Convert to lowercase and remove punctuation
            text = re.sub(r'[^\w\s]', '', text.lower())
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            processed_tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words
            ]
            
            return ' '.join(processed_tokens)
        except Exception as e:
            print(f"Error preprocessing text: {str(e)}")
            return text  # Return original text if processing fails

    def find_matching_diseases(self, symptoms):
        try:
            # Create a mask for matching symptoms
            mask = pd.DataFrame(False, index=self.intents_df.index, columns=['match'])
            
            for symptom in symptoms:
                symptom_match = self.intents_df[self.symptom_columns].apply(
                    lambda x: x.str.contains(symptom, case=False, na=False)
                ).any(axis=1)
                mask['match'] |= symptom_match
            
            matches = self.intents_df[mask['match']]
            print(f"Found {len(matches)} matches for symptoms: {symptoms}")
            return matches
            
        except Exception as e:
            print(f"Error in find_matching_diseases: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def generate_response(self, disease):
        try:
            # Get symptoms
            symptoms = [
                symptom for symptom in disease[self.symptom_columns]
                if pd.notna(symptom) and symptom.lower() != 'unknown'
            ]
            symptom_list = ', '.join(symptoms) if symptoms else ''

            # Get medications
            medication_columns = ['Medication_1', 'Medication_2', 'Medication_3', 'Medication_4']
            medications = [
                med for med in disease[medication_columns]
                if pd.notna(med) and med.lower() != 'unknown'
            ]
            medication_list = ', '.join(medications) if medications else ''

            # Build response
            response = [f"Based on your symptoms, it could be {disease['Disease']}."]
            
            if symptom_list:
                response.append(f"Common symptoms include: {symptom_list}.")
            
            if medication_list:
                response.append(f"Typical treatments might involve: {medication_list}.")
            
            response.append("Please note: This is not a definitive diagnosis. Consult a healthcare professional for proper medical advice.")
            
            return ' '.join(response)
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error analyzing your symptoms. Please try again or consult a healthcare professional."

    def chat(self):
        print("Healthcare Chatbot: Hello! How can I assist you today? Please describe your main symptom.")
        self.text_to_speech("Hello! How can I assist you today? Please describe your main symptom.")
        
        symptoms = []
        while True:
            user_input = input("Healthcare Chatbot: Please describe your main symptom: ")
            if user_input.lower() in ['no', 'nope', 'done']:
                break
            
            preprocessed_input = self.preprocess_text(user_input)
            symptoms.append(preprocessed_input)
            print(f"Healthcare Chatbot: Got it - '{user_input}'")
            print("Healthcare Chatbot: Any other symptoms? Say 'no' if you're finished.")
            self.text_to_speech("Any other symptoms?")
        
        if not symptoms:
            print("Healthcare Chatbot: No symptoms were provided. I can't make a prediction without any symptoms.")
            self.text_to_speech("No symptoms were provided. I can't make a prediction without any symptoms.")
            return

        print("Healthcare Chatbot: Thank you. Let me analyze your symptoms...")
        matching_diseases = self.find_matching_diseases(symptoms)
        
        if matching_diseases.empty:
            response = "I couldn't find any matching conditions for your symptoms. Please consult a healthcare professional for proper diagnosis."
        else:
            disease_match_count = matching_diseases[self.symptom_columns].apply(
                lambda x: x.str.contains('|'.join(symptoms), case=False, na=False).sum(), axis=1
            )
            best_match = matching_diseases.loc[disease_match_count.idxmax()]
            response = self.generate_response(best_match)
        
        print(f"Healthcare Chatbot: {response}")
        self.text_to_speech(response)

if __name__ == "__main__":
    try:
        chatbot = HealthcareChatbot()
        chatbot.chat()
    except Exception as e:
        print(f"An error occurred: {str(e)}")