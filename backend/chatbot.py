import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from spellchecker import SpellChecker
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_model import DiseasePredictor
import pyttsx3
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import re
import warnings
warnings.filterwarnings('ignore')

# Initialize text-to-speech engine
try:
    engine = pyttsx3.init()
except:
    print("Warning: Could not initialize text-to-speech engine. TTS functionality will be disabled.")
    engine = None

class MedicalChatbot:
    def __init__(self):
        # Load and preprocess the dataset
        self.df = pd.read_csv('Priority_wise_MedicalDataset - Sheet1 (1).csv')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize NLP components
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.spell_checker = SpellChecker()
        self.tokenizer = RegexpTokenizer(r'\w+')
        
        # Initialize ML model
        print("Initializing XGBoost model...")
        self.ml_model = DiseasePredictor('Priority_wise_MedicalDataset - Sheet1 (1).csv')
        
        # Set confidence thresholds
        self.high_confidence = 0.6
        self.medium_confidence = 0.3
        self.low_confidence = 0.1
        
        # Load medical terms and synonyms
        self.medical_terms = self.load_medical_terms()
        
        # Add priority weights
        self.priority_weights = {
            1: 1.5,    # First symptom (primary) gets 50% more weight
            2: 1.3,    # Second symptom gets 30% more weight
            3: 1.1,    # Third symptom gets 10% more weight
            # Rest of the symptoms get normal weight (1.0)
        }
        
        # Prepare the dataset
        self.prepare_data()
        
    def load_medical_terms(self):
        """Load medical terms and their synonyms with expanded dictionary"""
        return {
            "fever": ["pyrexia", "hyperthermia", "temperature", "febrile", "high temperature", "hot", "chills"],
            "headache": ["cephalgia", "migraine", "head pain", "head ache", "skull pain", "cranial pain"],
            "cough": ["tussis", "coughing", "hack", "dry cough", "wet cough", "persistent cough", "chest cough"],
            "fatigue": ["tiredness", "exhaustion", "lethargy", "weakness", "drowsiness", "lack of energy", "worn out"],
            "nausea": ["sickness", "queasiness", "upset stomach", "feeling sick", "stomach upset", "queasy feeling"],
            "pain": ["ache", "discomfort", "soreness", "distress", "tenderness", "sharp pain", "dull pain"],
            "vomiting": ["emesis", "throwing up", "vomit", "regurgitation", "being sick", "heaving"],
            "diarrhea": ["loose stools", "watery stools", "frequent bowel movements", "loose bowels", "runny stool"],
            "chest pain": ["angina", "chest discomfort", "chest tightness", "chest pressure", "thoracic pain"],
            "shortness of breath": ["dyspnea", "breathlessness", "difficulty breathing", "labored breathing", "short of breath"],
            "dizziness": ["vertigo", "lightheadedness", "giddiness", "feeling faint", "spinning sensation", "wooziness"],
            "abdominal pain": ["stomach pain", "belly pain", "gastric pain", "tummy ache", "stomach cramps", "gut pain"],
            "sore throat": ["pharyngitis", "throat pain", "throat ache", "painful throat", "throat soreness"],
            "runny nose": ["rhinorrhea", "nasal discharge", "nasal drip", "running nose", "nose dripping"],
            "muscle pain": ["myalgia", "muscle ache", "muscular pain", "body ache", "muscle soreness"],
            "joint pain": ["arthralgia", "joint ache", "articular pain", "bone pain", "joint soreness"],
            "rash": ["skin eruption", "dermatitis", "skin rash", "hives", "skin irritation", "skin outbreak"],
            "swelling": ["edema", "inflammation", "bloating", "puffiness", "enlarged", "swollen"],
            "bleeding": ["hemorrhage", "blood loss", "bleeding out", "blood flow", "bloody discharge"],
            "numbness": ["paresthesia", "tingling", "pins and needles", "loss of sensation", "numbness and tingling"],
            "anxiety": ["nervousness", "worry", "apprehension", "unease", "restlessness", "panic"],
            "depression": ["low mood", "sadness", "melancholy", "feeling down", "depressed mood"],
            "insomnia": ["sleeplessness", "inability to sleep", "sleep disorder", "trouble sleeping", "poor sleep"],
            "loss of appetite": ["anorexia", "poor appetite", "decreased appetite", "not hungry", "appetite loss"]
        }
        
    def prepare_data(self):
        # Combine all symptoms for each disease into a single string
        symptom_cols = [col for col in self.df.columns if col.startswith('Symptom_')]
        self.df['all_symptoms'] = self.df[symptom_cols].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocess_text,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit vectorizer on symptoms
        self.symptom_vectors = self.vectorizer.fit_transform(self.df['all_symptoms'])

    def correct_spelling(self, text):
        """Correct spelling while preserving medical terms"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            word_lower = word.lower()
            
            # Check if word is a medical term or its synonym
            is_medical_term = any(
                word_lower in [term.lower()] + [syn.lower() for syn in synonyms]
                for term, synonyms in self.medical_terms.items()
            )
            
            if is_medical_term:
                corrected_words.append(word)
            else:
                # Check if the word needs correction
                if word_lower not in self.spell_checker:
                    correction = self.spell_checker.correction(word)
                    corrected_words.append(correction if correction else word)
                else:
                    corrected_words.append(word)
        
        return ' '.join(corrected_words)

    def preprocess_text(self, text):
        """Preprocess text with simple tokenization"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Correct spelling
        text = self.correct_spelling(text)
        
        # Tokenize using simple word boundaries
        tokens = self.tokenizer.tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return tokens

    def process_symptoms(self, symptoms_list):
        """Process the list of symptoms with priority-based weighting and ML model predictions"""
        # Get ML model predictions
        ml_predictions = self.ml_model.predict_disease(symptoms_list)
        
        # First, try to find exact matches for diseases with few symptoms
        exact_matches = []
        for idx, row in self.df.iterrows():
            # Get all non-null symptoms for this disease
            disease_symptoms = [
                str(row[col]).lower() 
                for col in self.df.columns 
                if col.startswith('Symptom_') and pd.notna(row[col]) and row[col]
            ]
            
            # Convert user symptoms to lowercase for comparison
            user_symptoms_lower = [s.lower() for s in symptoms_list]
            
            # Check if this is a disease with few symptoms (1 or 2)
            if 1 <= len(disease_symptoms) <= 2:
                # Check if all user symptoms match this disease's symptoms
                user_symptoms_match = all(
                    any(user_symptom in disease_symptom or disease_symptom in user_symptom 
                        for disease_symptom in disease_symptoms)
                    for user_symptom in user_symptoms_lower
                )
                
                # Check if all disease symptoms are matched by user symptoms
                disease_symptoms_match = all(
                    any(user_symptom in disease_symptom or disease_symptom in user_symptom 
                        for user_symptom in user_symptoms_lower)
                    for disease_symptom in disease_symptoms
                )
                
                if user_symptoms_match and disease_symptoms_match and len(user_symptoms_lower) <= 2:
                    symptoms = []
                    for col in [c for c in row.index if c.startswith('Symptom_')]:
                        if pd.notna(row[col]) and row[col]:
                            priority = int(col.split('_')[1])
                            symptom_text = row[col].lower()
                            
                            user_priority = None
                            for i, user_symptom in enumerate(symptoms_list, 1):
                                if user_symptom.lower() in symptom_text or symptom_text in user_symptom.lower():
                                    user_priority = i
                                    break
                            
                            symptoms.append({
                                'symptom': row[col],
                                'priority': priority,
                                'user_priority': user_priority
                            })
                    
                    exact_matches.append({
                        'disease': row['Disease_Name'],
                        'category': row['Category'],
                        'severity': row['Severity_Level'],
                        'confidence': 0.95,  # High confidence for exact matches
                        'base_confidence': 0.95,
                        'ml_confidence': 0,
                        'match_quality': "High",
                        'symptoms': sorted(symptoms, key=lambda x: (x['user_priority'] or 999, x['priority'])),
                        'medications': [med for med in row[[c for c in row.index if c.startswith('Medication_')]]
                                      if pd.notna(med) and med],
                        'is_severe': row['Severity_Level'].lower() in ['high', 'severe'],
                        'is_high_confidence': True,
                        'is_medium_confidence': False
                    })
        
        # If we found exact matches for diseases with few symptoms, return those
        if exact_matches:
            return exact_matches
        
        # If no exact matches found, proceed with the regular matching process
        weighted_symptoms = []
        for i, symptom in enumerate(symptoms_list, 1):
            weight = self.priority_weights.get(i, 1.0)
            full_repeats = int(weight)
            partial_repeat = weight - full_repeats
            weighted_symptoms.extend([symptom] * full_repeats)
            if partial_repeat > 0:
                weighted_symptoms.append(symptom)
        
        user_input = " ".join(weighted_symptoms)
        processed_input = ' '.join(self.preprocess_text(user_input))
        input_vector = self.vectorizer.transform([processed_input])
        similarity_scores = cosine_similarity(input_vector, self.symptom_vectors).flatten()
        top_indices = similarity_scores.argsort()[-5:][::-1]
        
        best_match = None
        highest_confidence = 0
        
        # Combine ML predictions with traditional predictions
        for idx in top_indices:
            if similarity_scores[idx] > self.low_confidence:
                disease = self.df.iloc[idx]
                symptoms = []
                disease_symptoms = []
                
                for col in [c for c in disease.index if c.startswith('Symptom_')]:
                    if pd.notna(disease[col]) and disease[col]:
                        priority = int(col.split('_')[1])
                        symptom_text = disease[col].lower()
                        disease_symptoms.append(symptom_text)
                        
                        user_priority = None
                        for i, user_symptom in enumerate(symptoms_list, 1):
                            if user_symptom.lower() in symptom_text or symptom_text in user_symptom.lower():
                                user_priority = i
                                break
                        
                        symptoms.append({
                            'symptom': disease[col],
                            'priority': priority,
                            'user_priority': user_priority
                        })
                
                base_confidence = similarity_scores[idx]
                
                # Boost confidence if ML model also predicted this disease
                ml_boost = 0
                for ml_pred in ml_predictions:
                    if ml_pred['disease'] == disease['Disease_Name']:
                        ml_boost = ml_pred['probability'] * 0.3  # 30% boost from ML prediction
                        break
                
                primary_symptom_bonus = 0
                for i, user_symptom in enumerate(symptoms_list[:3], 1):
                    if any(user_symptom.lower() in s for s in disease_symptoms):
                        primary_symptom_bonus += self.priority_weights.get(i, 1.0) - 1
                
                adjusted_confidence = min(1.0, base_confidence * (1 + primary_symptom_bonus * 0.2) + ml_boost)
                
                # Determine disease severity and confidence level
                is_severe = disease['Severity_Level'].lower() in ['high', 'severe']
                is_high_confidence = adjusted_confidence >= 0.7
                is_medium_confidence = 0.5 <= adjusted_confidence < 0.7
                
                if (is_high_confidence and is_severe) or \
                   (is_medium_confidence and not is_severe) or \
                   (adjusted_confidence > highest_confidence):
                    highest_confidence = adjusted_confidence
                    
                    # Calculate match quality
                    if adjusted_confidence >= self.high_confidence:
                        match_quality = "High"
                    elif adjusted_confidence >= self.medium_confidence:
                        match_quality = "Medium"
                    else:
                        match_quality = "Low"
                    
                    best_match = {
                        'disease': disease['Disease_Name'],
                        'category': disease['Category'],
                        'severity': disease['Severity_Level'],
                        'confidence': adjusted_confidence,
                        'base_confidence': base_confidence,
                        'ml_confidence': ml_boost / 0.3 if ml_boost > 0 else 0,  # Convert boost back to original confidence
                        'match_quality': match_quality,
                        'symptoms': sorted(symptoms, key=lambda x: (x['user_priority'] or 999, x['priority'])),
                        'medications': [med for med in disease[[c for c in disease.index if c.startswith('Medication_')]]
                                      if pd.notna(med) and med],
                        'is_severe': is_severe,
                        'is_high_confidence': is_high_confidence,
                        'is_medium_confidence': is_medium_confidence
                    }
        
        return [best_match] if best_match else []

    def generate_response(self, results):
        """Generate response with Hindi translation and text-to-speech"""
        if not results or not results[0]:
            return "I couldn't find any matching conditions based on the symptoms you described. Could you please provide more specific symptoms?"
        
        result = results[0]  # Get the single best match
        
        # Format the basic response
        response = f"Based on your symptoms You may have:\n\n"
        response += f"Diagnosis: {result['disease']}\n"
        response += f"Category: {result['category']}\n"
        response += f"Severity Level: {result['severity']}\n\n"
        
        # Add symptoms
        response += "Symptoms:\n"
        seen_symptoms = set()
        for symptom in result['symptoms']:
            symptom_text = symptom['symptom']
            if symptom_text not in seen_symptoms:
                seen_symptoms.add(symptom_text)
                response += f"• {symptom_text}\n"
        
        # Add medications if available
        if result['medications']:
            response += "\nRecommended Medications:\n"
            for med in result['medications']:
                if pd.notna(med):
                    response += f"• {med}\n"
        
        # Store English response
        english_response = response
        
        # Translate to Hindi using indic-transliteration
        try:
            # Convert English text to Devanagari (Hindi script)
            hindi_translation = transliterate(english_response, sanscript.ITRANS, sanscript.DEVANAGARI)
            response += "\nहिंदी अनुवाद (Hindi Translation):\n" + hindi_translation
        except Exception as e:
            print(f"Translation error: {str(e)}")
        
        # Text to speech for English response only
        try:
            engine.say(english_response)
            engine.runAndWait()
        except Exception as e:
            print(f"Text-to-speech error: {str(e)}")
        
        return response

def main():
    chatbot = MedicalChatbot()
    print("Hello! How can I assist you? Can you tell me your symptoms one by one?")
    engine.say("Hello! How can I assist you? Can you tell me your symptoms one by one?")
    engine.runAndWait()
    
    while True:
        symptoms_list = []
        
        while True:
            user_input = input("\nEnter symptom: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nTake care! Remember to consult with healthcare professionals for medical advice.")
                return
            
            if not user_input:
                print("Please enter a valid symptom.")
                continue
                
            if user_input.lower() == 'no':
                if not symptoms_list:
                    print("Please enter at least one symptom.")
                    continue
                break
            
            # Correct spelling before adding to the list
            corrected_input = chatbot.correct_spelling(user_input)
            if corrected_input != user_input:
                print(f"Corrected '{user_input}' to '{corrected_input}'")
            symptoms_list.append(corrected_input)
            
            prompt = "Got it! Any other symptoms? (say no if you're finished)"
            print(prompt)
            engine.say(prompt)
            engine.runAndWait()
        
        print("\nAnalyzing your symptoms...")
        # Process all collected symptoms
        results = chatbot.process_symptoms(symptoms_list)
        response = chatbot.generate_response(results)
        print(response)
        
        # Ask if user wants to start over
        while True:
            another = input("\nWould you like to check different symptoms? (yes/no): ").strip().lower()
            if another in ['yes', 'no']:
                break
            print("Please answer 'yes' or 'no'")
        
        if another == 'no':
            print("\nTake care! Remember to consult with healthcare professionals for medical advice.")
            break

if __name__ == "__main__":
    main()