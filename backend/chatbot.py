import pyttsx3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import os
import traceback
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
import numpy as np

class HealthcareChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Use absolute path for the dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, 'MedDatasetFinal_modeled.csv')
        self.intents_df = self.load_intents(dataset_path)
        self.symptom_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
        
        # Initialize symptom importance weights
        self.symptom_weights = {
            'Symptom_1': 1.0,  # Primary symptom gets full weight
            'Symptom_2': 0.8,  # Secondary symptoms get slightly lower weights
            'Symptom_3': 0.6,
            'Symptom_4': 0.4
        }
        
        # Create symptom similarity cache
        self.similarity_cache = {}

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
            text = text.lower()
            
            # Keep hyphens for medical terms (e.g., "non-specific")
            text = re.sub(r'[^\w\s-]', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Custom stopwords for medical context
            medical_stopwords = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'a', 'an', 'is', 'are', 'was', 'were'}
            self.stop_words = self.stop_words - medical_stopwords
            
            # Remove stopwords and lemmatize, but keep medical terms intact
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words or '-' in token:  # Keep hyphenated terms
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
            
            processed_text = ' '.join(processed_tokens)
            print(f"Preprocessed text: '{text}' -> '{processed_text}'")
            return processed_text
            
        except Exception as e:
            print(f"Error preprocessing text: {str(e)}")
            return text  # Return original text if processing fails

    def calculate_symptom_similarity(self, symptom1, symptom2):
        """Calculate similarity between two symptoms using multiple metrics."""
        if not symptom1 or not symptom2:
            return 0.0
            
        cache_key = tuple(sorted([str(symptom1).lower(), str(symptom2).lower()]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Convert to lowercase strings
        s1, s2 = str(symptom1).lower(), str(symptom2).lower()
        
        # Calculate different similarity metrics
        ratio = fuzz.ratio(s1, s2) / 100.0
        partial_ratio = fuzz.partial_ratio(s1, s2) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(s1, s2) / 100.0
        sequence_ratio = SequenceMatcher(None, s1, s2).ratio()
        
        # Combine metrics with weights
        similarity = (ratio * 0.3 + 
                     partial_ratio * 0.3 + 
                     token_sort_ratio * 0.2 + 
                     sequence_ratio * 0.2)
        
        self.similarity_cache[cache_key] = similarity
        return similarity

    def find_matching_diseases(self, symptoms):
        try:
            print(f"Searching for symptoms: {symptoms}")
            
            # Calculate similarity scores for each disease
            disease_scores = []
            
            for idx, row in self.intents_df.iterrows():
                disease_score = 0.0
                matched_symptoms = set()
                
                # For each user symptom
                for user_symptom in symptoms:
                    best_symptom_match = 0.0
                    
                    # Compare with each disease symptom
                    for col in self.symptom_columns:
                        disease_symptom = row[col]
                        if pd.isna(disease_symptom):
                            continue
                            
                        # Calculate similarity
                        similarity = self.calculate_symptom_similarity(user_symptom, disease_symptom)
                        
                        # Apply column weight
                        weighted_similarity = similarity * self.symptom_weights[col]
                        
                        if weighted_similarity > best_symptom_match:
                            best_symptom_match = weighted_similarity
                    
                    disease_score += best_symptom_match
                
                # Normalize score by number of symptoms
                if symptoms:
                    disease_score /= len(symptoms)
                
                disease_scores.append({
                    'index': idx,
                    'disease': row['Disease'],
                    'score': disease_score
                })
            
            # Sort diseases by score
            disease_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Filter diseases with score above threshold
            threshold = 0.3  # Adjust this threshold as needed
            matching_indices = [d['index'] for d in disease_scores if d['score'] > threshold]
            
            if not matching_indices:
                return pd.DataFrame()
            
            matches = self.intents_df.loc[matching_indices].copy()
            
            # Add scores to the matches
            matches['match_score'] = [d['score'] for d in disease_scores if d['score'] > threshold]
            
            print(f"Found {len(matches)} matches with scores above {threshold}:")
            for _, match in matches.iterrows():
                print(f"Disease: {match['Disease']}, Score: {match['match_score']:.2f}")
            
            return matches.sort_values('match_score', ascending=False)
            
        except Exception as e:
            print(f"Error in find_matching_diseases: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def generate_response(self, disease):
        try:
            # Get symptoms with confidence scores
            match_score = disease.get('match_score', 0)
            confidence_level = "high" if match_score > 0.7 else "moderate" if match_score > 0.5 else "low"
            
            symptoms = [
                symptom for symptom in disease[self.symptom_columns]
                if pd.notna(symptom) and symptom.lower() != 'unknown'
            ]
            symptom_list = ', '.join(symptoms) if symptoms else ''

            medication_columns = ['Medication_1', 'Medication_2', 'Medication_3', 'Medication_4']
            medications = [
                med for med in disease[medication_columns]
                if pd.notna(med) and med.lower() != 'unknown'
            ]
            medication_list = ', '.join(medications) if medications else ''

            # Build response with confidence level
            response = [
                f"Based on your symptoms, there is a {confidence_level} likelihood that it could be {disease['Disease']}."
            ]
            
            if symptom_list:
                response.append(f"Common symptoms include: {symptom_list}.")
            
            if medication_list:
                response.append(f"Typical treatments might involve: {medication_list}.")
            
            if confidence_level != "high":
                response.append("Given the symptom match confidence level, please consider alternative conditions as well.")
                
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