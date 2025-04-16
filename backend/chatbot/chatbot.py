import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from spellchecker import SpellChecker
from ml_model import DiseasePredictor
import pyttsx3
from googletrans import Translator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import re
import warnings
warnings.filterwarnings('ignore')

class MedicalChatbot:
    def __init__(self, language='english'):
        # Set language preference
        self.language = language.lower()
        self.translator = Translator()
        
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
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        # Configure the TTS engine
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Get available voices and set a voice
        voices = self.tts_engine.getProperty('voices')
        if voices:  # If voices are available, select a voice
            # Try to select a female voice if available
            female_voices = [v for v in voices if v.gender == 'female']
            if female_voices:
                self.tts_engine.setProperty('voice', female_voices[0].id)
            else:
                # Otherwise, use the first available voice
                self.tts_engine.setProperty('voice', voices[0].id)
        
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
        
        # Hindi-English translations for common phrases
        self.translations = {
            'enter_symptom': {
                'english': 'Please enter a symptom.',
                'hindi': 'कृपया एक लक्षण दर्ज करें।'
            },
            'enter_another': {
                'english': 'Enter another symptom or type \'no\' to get diagnosis.',
                'hindi': 'एक और लक्षण दर्ज करें या निदान प्राप्त करने के लिए \'no\' टाइप करें।'
            },
            'farewell': {
                'english': 'Take care! Remember to consult with healthcare professionals for medical advice.',
                'hindi': 'अपना ख्याल रखें! चिकित्सा सलाह के लिए स्वास्थ्य पेशेवरों से परामर्श करना याद रखें।'
            },
            'valid_symptom': {
                'english': 'Please enter a valid symptom.',
                'hindi': 'कृपया एक वैध लक्षण दर्ज करें।'
            },
            'at_least_one': {
                'english': 'Please enter at least one symptom.',
                'hindi': 'कृपया कम से कम एक लक्षण दर्ज करें।'
            },
            'current_symptoms': {
                'english': 'Current symptoms',
                'hindi': 'वर्तमान लक्षण'
            },
            'analyzing': {
                'english': 'Analyzing symptoms. Please wait a moment.',
                'hindi': 'लक्षणों का विश्लेषण किया जा रहा है। कृपया एक क्षण प्रतीक्षा करें।'
            },
            'check_different': {
                'english': 'Would you like to check different symptoms? Say yes or no.',
                'hindi': 'क्या आप अलग-अलग लक्षणों की जांच करना चाहेंगे? हां या नहीं कहें।'
            },
            'yes_no': {
                'english': 'Please answer yes or no',
                'hindi': 'कृपया हां या नहीं में जवाब दें'
            },
            'welcome': {
                'english': 'Hello! I\'m here to help you understand potential medical conditions.',
                'hindi': 'नमस्ते! मैं आपको संभावित चिकित्सा स्थितियों को समझने में मदद करने के लिए यहां हूं।'
            },
            'corrected': {
                'english': 'Corrected',
                'hindi': 'सुधारा गया'
            },
            'to': {
                'english': 'to',
                'hindi': 'से'
            }
        }
        
        # Prepare the dataset
        self.prepare_data()
        
    def speak(self, text):
        """Text-to-speech function to read out text"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def translate_text(self, text, target_language='hindi'):
        """Translate text to the target language"""
        try:
            if target_language.lower() == 'english':
                return text
            translation = self.translator.translate(text, dest='hi' if target_language.lower() == 'hindi' else target_language)
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def transliterate_to_devanagari(self, text):
        """Transliterate Latin text to Devanagari script for Hindi"""
        try:
            transliterated = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
            return transliterated
        except Exception as e:
            print(f"Transliteration error: {e}")
            return text
            
    def get_phrase(self, key):
        """Get a phrase in the current language"""
        if key in self.translations:
            return self.translations[key].get(self.language, self.translations[key]['english'])
        return key  # Return the key itself if not found
    
    def set_language(self, language):
        """Change the language of the chatbot"""
        self.language = language.lower()
        # Adjust TTS settings for Hindi if needed
        if self.language == 'hindi':
            # Lower the speech rate for Hindi
            self.tts_engine.setProperty('rate', 130)
        else:
            # Reset to default for English
            self.tts_engine.setProperty('rate', 150)
            
    def process_user_input(self, user_input):
        """Process user input based on language settings"""
        # For Hindi input, first try to convert to English for processing
        if self.language == 'hindi':
            try:
                # Translate Hindi input to English for processing
                english_input = self.translator.translate(user_input, src='hi', dest='en').text
                return english_input
            except:
                # If translation fails, try to process as is
                return user_input
        return user_input
        
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
        """Generate enhanced response with priority-based information and ML insights"""
        if not results or not results[0]:
            if self.language == 'hindi':
                return "मैं आपके बताए गए लक्षणों के आधार पर कोई मिलान वाली स्थिति नहीं ढूंढ सका। कृपया अधिक विशिष्ट लक्षण प्रदान करें?"
            else:
                return "I couldn't find any matching conditions based on the symptoms you described. Could you please provide more specific symptoms?"
        
        result = results[0]  # Get the single best match
        response_parts = []
        
        # Determine the type of warning based on severity and confidence
        if result['is_severe'] and result['is_high_confidence']:
            if self.language == 'hindi':
                response_parts.append("\n⚠️ अत्यावश्यक: उच्च विश्वास गंभीर स्थिति मिलान ⚠️")
                response_parts.append("आपके लक्षणों के आधार पर, एक गंभीर स्थिति का मजबूत संकेत है।")
                response_parts.append("कृपया तुरंत चिकित्सा सहायता लें!\n")
            else:
                response_parts.append("\n⚠️ URGENT: High Confidence Severe Condition Match ⚠️")
                response_parts.append("Based on your symptoms, there is a strong indication of a severe condition.")
                response_parts.append("Please seek immediate medical attention!\n")
        elif result['is_medium_confidence']:
            if self.language == 'hindi':
                response_parts.append("\n🟡 मध्यम विश्वास मिलान")
                response_parts.append("आपके लक्षणों के आधार पर, इस स्थिति के लिए ध्यान देने की आवश्यकता है लेकिन यह गंभीर नहीं हो सकती है।\n")
            else:
                response_parts.append("\n🟡 Moderate Confidence Match")
                response_parts.append("Based on your symptoms, this condition requires attention but may not be severe.\n")
        else:
            if self.language == 'hindi':
                response_parts.append("\nआपके लक्षणों के आधार पर, यहां सबसे संभावित स्थिति है:")
            else:
                response_parts.append("\nBased on your symptoms, here is the most likely condition:")
        
        # Format the disease information
        confidence_percent = round(result['confidence'] * 100, 1)
        base_confidence_percent = round(result['base_confidence'] * 100, 1)
        ml_confidence_percent = round(result['ml_confidence'] * 100, 1)
        
        # Disease and Confidence Information
        disease_name = result['disease']
        if self.language == 'hindi':
            try:
                disease_name = self.translate_text(disease_name)
            except:
                pass  # Keep original if translation fails
            
            response_parts.append(f"\n📊 निदान: {disease_name}")
            response_parts.append(f"मिलान गुणवत्ता: {self.translate_text(result['match_quality'])} ({confidence_percent}% विश्वास)")
            response_parts.append(f"• आधार समानता: {base_confidence_percent}%")
            if result['ml_confidence'] > 0:
                response_parts.append(f"• ML मॉडल विश्वास: {ml_confidence_percent}%")
                response_parts.append("↑ मशीन लर्निंग भविष्यवाणियों द्वारा अंतिम विश्वास बढ़ाया गया")
        else:
            response_parts.append(f"\n📊 Diagnosis: {disease_name}")
            response_parts.append(f"Match Quality: {result['match_quality']} ({confidence_percent}% confidence)")
            response_parts.append(f"• Base similarity: {base_confidence_percent}%")
            if result['ml_confidence'] > 0:
                response_parts.append(f"• ML model confidence: {ml_confidence_percent}%")
                response_parts.append("↑ Final confidence boosted by machine learning predictions")
        
        # Disease Category and Severity
        category = result['category']
        severity = result['severity']
        
        if self.language == 'hindi':
            try:
                category = self.translate_text(category)
                severity = self.translate_text(severity)
            except:
                pass  # Keep original if translation fails
                
            response_parts.append(f"\nश्रेणी: {category}")
            if severity.lower() in ['high', 'severe', 'उच्च', 'गंभीर']:
                response_parts.append(f"गंभीरता स्तर: ⚠️ {severity.upper()} ⚠️")
            else:
                response_parts.append(f"गंभीरता स्तर: {severity}")
        else:
            response_parts.append(f"\nCategory: {category}")
            if severity.lower() in ['high', 'severe']:
                response_parts.append(f"Severity Level: ⚠️ {severity.upper()} ⚠️")
            else:
                response_parts.append(f"Severity Level: {severity}")
        
        # Show symptoms with priority information (remove duplicates)
        if self.language == 'hindi':
            response_parts.append("\nमिलान लक्षण:")
        else:
            response_parts.append("\nMatching Symptoms:")
            
        seen_symptoms = set()  # To track unique symptoms
        for symptom in result['symptoms']:
            symptom_text = symptom['symptom']
            if symptom_text not in seen_symptoms:
                seen_symptoms.add(symptom_text)
                
                # Translate symptom if in Hindi mode
                if self.language == 'hindi':
                    try:
                        symptom_text = self.translate_text(symptom_text)
                    except:
                        pass  # Keep original if translation fails
                
                priority_label = ""
                if symptom['user_priority']:
                    if symptom['user_priority'] == 1:
                        priority_label = "🔴 " + ("प्राथमिक" if self.language == 'hindi' else "PRIMARY")
                    elif symptom['user_priority'] == 2:
                        priority_label = "🟡 " + ("द्वितीयक" if self.language == 'hindi' else "SECONDARY")
                    elif symptom['user_priority'] == 3:
                        priority_label = "🟢 " + ("तृतीयक" if self.language == 'hindi' else "TERTIARY")
                
                response_parts.append(f"• {symptom_text} {priority_label}")
        
        # Show medications if available
        if result['medications']:
            if self.language == 'hindi':
                response_parts.append("\nअनुशंसित दवाएं:")
            else:
                response_parts.append("\nRecommended Medications:")
                
            for med in result['medications']:
                if pd.notna(med):  # Check if medication is not NaN
                    med_text = med
                    # Translate medication if in Hindi mode
                    if self.language == 'hindi':
                        try:
                            med_text = self.translate_text(med_text)
                        except:
                            pass  # Keep original if translation fails
                    response_parts.append(f"• {med_text}")
        
        return "\n".join(response_parts)

def main():
    # Initially start with English
    language = 'english'
    print("Starting in English mode. Type 'switch' at any time to toggle between English and Hindi.")
    
    chatbot = MedicalChatbot(language)
    
    welcome_message = chatbot.get_phrase('welcome')
    print(welcome_message)
    chatbot.speak(welcome_message)
    
    while True:
        symptoms_list = []
        
        while True:
            if not symptoms_list:
                prompt = "\n🔍 " + (chatbot.get_phrase('enter_symptom') if chatbot.language == 'hindi' else "Please enter a symptom: ")
                print(prompt, end="")
                chatbot.speak(chatbot.get_phrase('enter_symptom'))
                user_input = input().strip()
            else:
                prompt = "\n🔍 " + (chatbot.get_phrase('enter_another') if chatbot.language == 'hindi' else "Enter another symptom (or type 'no' to get diagnosis): ")
                print(prompt, end="")
                chatbot.speak(chatbot.get_phrase('enter_another'))
                user_input = input().strip()
            
            # Check for language switch command
            if user_input.lower() == 'switch':
                if chatbot.language == 'english':
                    chatbot.set_language('hindi')
                    switch_msg = "स्विच किया गया: अब हिंदी में बात करें"
                    print(switch_msg)
                    chatbot.speak(switch_msg)
                else:
                    chatbot.set_language('english')
                    switch_msg = "Switched: Now speaking in English"
                    print(switch_msg)
                    chatbot.speak(switch_msg)
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                farewell = "\n👋 " + chatbot.get_phrase('farewell')
                print(farewell)
                chatbot.speak(chatbot.get_phrase('farewell'))
                return
            
            if not user_input:
                message = "❌ " + chatbot.get_phrase('valid_symptom')
                print(message)
                chatbot.speak(chatbot.get_phrase('valid_symptom'))
                continue
                
            if user_input.lower() == 'no':
                if not symptoms_list:
                    message = "❌ " + chatbot.get_phrase('at_least_one')
                    print(message)
                    chatbot.speak(chatbot.get_phrase('at_least_one'))
                    continue
                break
            
            # Process user input based on language
            processed_input = chatbot.process_user_input(user_input)
            
            # Correct spelling before adding to the list
            corrected_input = chatbot.correct_spelling(processed_input)
            if corrected_input != processed_input:
                correction_msg = f"✏️ {chatbot.get_phrase('corrected')} '{user_input}' {chatbot.get_phrase('to')} '{corrected_input}'"
                print(correction_msg)
                chatbot.speak(f"{chatbot.get_phrase('corrected')} {user_input} {chatbot.get_phrase('to')} {corrected_input}")
            
            symptoms_list.append(corrected_input)
            symptoms_text = ', '.join(symptoms_list)
            
            # Display in current language
            if chatbot.language == 'hindi':
                try:
                    symptoms_hindi = chatbot.translate_text(symptoms_text)
                    symptoms_msg = f"📋 {chatbot.get_phrase('current_symptoms')}: {symptoms_hindi}"
                except:
                    symptoms_msg = f"📋 {chatbot.get_phrase('current_symptoms')}: {symptoms_text}"
            else:
                symptoms_msg = f"📋 Current symptoms: {symptoms_text}"
                
            print(symptoms_msg)
            chatbot.speak(symptoms_msg if chatbot.language == 'english' else chatbot.get_phrase('current_symptoms') + ": " + symptoms_text)
        
        analyzing_msg = "\n🔄 " + chatbot.get_phrase('analyzing')
        print(analyzing_msg)
        
        # Process all collected symptoms
        results = chatbot.process_symptoms(symptoms_list)
        response = chatbot.generate_response(results)
        print("\n📊 Analysis Results:", response)
        
        # Extract and speak the essential parts of the analysis
        if results and results[0]:
            result = results[0]
            if chatbot.language == 'english':
                speech_response = f"Based on your symptoms, the most likely diagnosis is {result['disease']} with {round(result['confidence'] * 100, 1)} percent confidence. "
                
                if result['is_severe'] and result['is_high_confidence']:
                    speech_response += "This appears to be a severe condition. Please seek immediate medical attention! "
                
                if result['medications']:
                    medications = [med for med in result['medications'] if pd.notna(med)]
                    if medications:
                        speech_response += f"Recommended medications include: {', '.join(medications)}. "
                
                speech_response += "Remember, this is for educational purposes only. Always consult healthcare professionals."
            else:
                # Translate key parts to Hindi
                try:
                    disease_hindi = chatbot.translate_text(result['disease'])
                    speech_response = f"आपके लक्षणों के आधार पर, सबसे संभावित निदान {disease_hindi} है, {round(result['confidence'] * 100, 1)} प्रतिशत विश्वास के साथ। "
                    
                    if result['is_severe'] and result['is_high_confidence']:
                        speech_response += "यह एक गंभीर स्थिति प्रतीत होती है। कृपया तुरंत चिकित्सा सहायता लें! "
                    
                    if result['medications']:
                        medications = [med for med in result['medications'] if pd.notna(med)]
                        if medications:
                            meds_hindi = chatbot.translate_text(', '.join(medications))
                            speech_response += f"अनुशंसित दवाएं इसमें शामिल हैं: {meds_hindi}। "
                    
                    speech_response += "याद रखें, यह केवल शैक्षिक उद्देश्यों के लिए है। हमेशा स्वास्थ्य पेशेवरों से परामर्श करें।"
                except:
                    # Fall back to English if translation fails
                    disease_name = result['disease']
                    speech_response = f"Based on your symptoms, the most likely diagnosis is {disease_name} with {round(result['confidence'] * 100, 1)} percent confidence."
            
            chatbot.speak(speech_response)
        else:
            no_match_response = "I couldn't find any matching conditions based on the symptoms you described. Could you please provide more specific symptoms?"
            if chatbot.language == 'hindi':
                try:
                    no_match_response = chatbot.translate_text(no_match_response)
                except:
                    pass
            chatbot.speak(no_match_response)
        
        # Ask if user wants to start over
        while True:
            print("\n🔄 " + (chatbot.get_phrase('check_different') if chatbot.language == 'hindi' else "Would you like to check different symptoms? (yes/no): "), end="")
            chatbot.speak(chatbot.get_phrase('check_different'))
            another = input().strip().lower()
            
            # Check for language switch
            if another.lower() == 'switch':
                if chatbot.language == 'english':
                    chatbot.set_language('hindi')
                    switch_msg = "स्विच किया गया: अब हिंदी में बात करें"
                    print(switch_msg)
                    chatbot.speak(switch_msg)
                else:
                    chatbot.set_language('english')
                    switch_msg = "Switched: Now speaking in English"
                    print(switch_msg)
                    chatbot.speak(switch_msg)
                continue
                
            if another in ['yes', 'no', 'हां', 'नहीं']:
                break
            error_msg = "❌ " + chatbot.get_phrase('yes_no')
            print(error_msg)
            chatbot.speak(chatbot.get_phrase('yes_no'))
        
        if another.lower() in ['no', 'नहीं']:
            farewell = "\n👋 " + chatbot.get_phrase('farewell')
            print(farewell)
            chatbot.speak(chatbot.get_phrase('farewell'))
            break

if __name__ == "__main__":
    main()