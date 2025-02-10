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
from translate import Translator

class HealthcareChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.translator_en_to_hi = Translator(to_lang='hi', from_lang='en')
        self.translator_hi_to_en = Translator(to_lang='en', from_lang='hi')
        
        # Use absolute path for the dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, 'MedDatasetFinal_modeled.csv')
        self.intents_df = self.load_intents(dataset_path)
        self.symptom_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
        
        # Add conversation patterns
        self.greeting_patterns = {
            'greetings': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 
                'good evening', 'namaste', 'hola', 'whats up', "what's up",
                'howdy', 'greetings', 'sup'
            ],
            'goodbyes': [
                'bye', 'goodbye', 'see you', 'take care', 'farewell',
                'have a good day', 'until next time', 'catch you later'
            ],
            'thanks': [
                'thank you', 'thanks', 'appreciate it', 'thank you so much',
                'thanks a lot', 'grateful', 'thanking you'
            ],
            'how_are_you': [
                'how are you', 'how do you do', 'how are you doing',
                'how have you been', 'whats going on', "what's going on"
            ],
            'help_requests': [
                'help', 'can you help', 'need help', 'assist', 'support',
                'guidance', 'what can you do', 'how can you help'
            ]
        }
        
        # Add response templates
        self.response_templates = {
            'greetings': [
                "Hello! I'm your healthcare assistant. How can I help you today?",
                "Hi there! I'm here to help with your health concerns. What symptoms are you experiencing?",
                "Welcome! I'm your medical chatbot. How may I assist you?"
            ],
            'goodbyes': [
                "Take care! Remember to consult a healthcare professional for proper medical advice.",
                "Goodbye! Stay healthy and don't hesitate to return if you need more assistance.",
                "Have a great day! Remember, I'm here if you need help understanding your symptoms."
            ],
            'thanks': [
                "You're welcome! Remember, I'm here to help understand symptoms, but always consult a doctor for proper diagnosis.",
                "Happy to help! Don't forget to seek professional medical advice for accurate diagnosis.",
                "Glad I could assist! Remember to verify any information with healthcare professionals."
            ],
            'how_are_you': [
                "I'm functioning well and ready to help you with any health concerns. What symptoms are you experiencing?",
                "I'm here and ready to assist you. What health issues would you like to discuss?",
                "I'm operational and prepared to help. What medical concerns can I address for you?"
            ],
            'help_requests': [
                "I can help you understand your symptoms and provide general health information. What symptoms are you experiencing?",
                "I'm designed to help identify possible conditions based on your symptoms. What symptoms would you like to discuss?",
                "I can assist by analyzing your symptoms and suggesting possible conditions. What symptoms are you having?"
            ]
        }
        
        # Initialize symptom importance weights
        self.symptom_weights = {
            'Symptom_1': 1.0,  # Primary symptom gets full weight
            'Symptom_2': 0.8,  # Secondary symptoms get slightly lower weights
            'Symptom_3': 0.6,
            'Symptom_4': 0.4
        }
        
        # Define common symptoms that should be weighted less to avoid over-matching with serious conditions
        self.common_symptoms = {
            'fever', 'headache', 'cough', 'fatigue', 'nausea', 'vomiting', 
            'diarrhea', 'sore throat', 'runny nose', 'muscle pain', 'weakness',
            'dizziness', 'cold', 'tiredness', 'body ache', 'chills'
        }
        
        # Define severity levels for diseases to prioritize common conditions for common symptoms
        self.disease_severity = {
            'low': ['Common Cold', 'Flu', 'Gastroenteritis', 'Migraine', 'Allergic Rhinitis'],
            'medium': ['Bronchitis', 'Sinusitis', 'Tonsillitis', 'Urinary Tract Infection'],
            'high': ['Pneumonia', 'Meningitis', 'Sepsis', 'Cancer', 'Heart Attack']
        }
        
        # Create symptom similarity cache
        self.similarity_cache = {}

    def load_intents(self, file_path):
        df = pd.read_csv(file_path)
        print("Columns in the CSV file:", df.columns)
        print("First few rows of the dataframe:")
        print(df.head())
        return df

    def text_to_speech(self, text, language='en'):
        try:
            if not hasattr(self, 'tts_engine'):
                self.tts_engine = pyttsx3.init()
                
            # Configure voice properties
            voices = self.tts_engine.getProperty('voices')
            
            # Set voice based on language
            if language == 'hi':
                # Try to find a Hindi voice
                hindi_voice = next((voice for voice in voices if 'hi' in voice.id.lower()), None)
                if hindi_voice:
                    self.tts_engine.setProperty('voice', hindi_voice.id)
                else:
                    # If no Hindi voice found, use default voice but adjust properties
                    self.tts_engine.setProperty('rate', 145)  # Slower rate for Hindi
                    self.tts_engine.setProperty('pitch', 100)  # Normal pitch
            else:
                # Use default English voice
                english_voice = next((voice for voice in voices if 'en' in voice.id.lower()), voices[0])
                self.tts_engine.setProperty('voice', english_voice.id)
                self.tts_engine.setProperty('rate', 175)  # Normal rate for English
            
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error in text_to_speech: {str(e)}")
            pass

    def translate_text(self, text, source_lang='hi', target_lang='en'):
        """Translate text between Hindi and English"""
        try:
            if not text or source_lang == target_lang:
                return text

            if source_lang == 'hi' and target_lang == 'en':
                translator = self.translator_hi_to_en
            else:
                translator = self.translator_en_to_hi

            # Split long text into sentences to handle translation limits
            sentences = text.split('.')
            translated_sentences = []
            
            for sentence in sentences:
                if sentence.strip():
                    try:
                        translated = translator.translate(sentence.strip())
                        translated_sentences.append(translated)
                    except Exception as e:
                        print(f"Error translating sentence: {str(e)}")
                        translated_sentences.append(sentence)
            
            return '. '.join(translated_sentences)
            
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text

    def preprocess_text(self, text, language='en'):
        """Preprocess text with language support"""
        try:
            print(f"Input text: {text}, Language: {language}")
            
            if not text:
                return text

            # For Hindi input
            if language == 'hi':
                # First store the original Hindi text
                original_hindi = text
                
                # Check if the text is already in Devanagari script
                is_devanagari = any('\u0900' <= c <= '\u097f' for c in text)
                
                if not is_devanagari:
                    # If text is Hindi written in English (transliteration), convert it to Devanagari
                    try:
                        from indic_transliteration import sanscript
                        from indic_transliteration.sanscript import transliterate
                        text = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
                        original_hindi = text  # Update original Hindi text with Devanagari version
                    except ImportError:
                        print("indic_transliteration not installed, skipping transliteration")
                
                # Translate to English for processing
                translated_text = self.translate_text(text, source_lang='hi', target_lang='en')
                print(f"Translated from Hindi: {translated_text}")
                
                # Process the English translation
                text = translated_text.lower()
                text = re.sub(r'[^\w\s-]', '', text)
                
                return {
                    'original': original_hindi,
                    'processed': text
                }
            
            # For English input, continue with normal processing
            text = text.lower()
            text = re.sub(r'[^\w\s-]', '', text)
            
            tokens = word_tokenize(text)
            
            # Custom stopwords for medical context
            medical_stopwords = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'a', 'an', 'is', 'are', 'was', 'were'}
            self.stop_words = self.stop_words - medical_stopwords
            
            # Remove stopwords and lemmatize
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words or '-' in token:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
            
            processed_text = ' '.join(processed_tokens)
            print(f"Preprocessed text: '{text}' -> '{processed_text}'")
            return processed_text
            
        except Exception as e:
            print(f"Error preprocessing text: {str(e)}")
            return text

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
            
            # Handle Hindi symptoms
            processed_symptoms = []
            for symptom in symptoms:
                if isinstance(symptom, dict) and 'processed' in symptom:
                    processed_symptoms.append(symptom['processed'])
                else:
                    processed_symptoms.append(symptom)
            
            print(f"Processed symptoms for matching: {processed_symptoms}")
            
            # Calculate similarity scores for each disease
            disease_scores = []
            
            # Count how many symptoms are common
            common_symptom_count = sum(1 for symptom in processed_symptoms 
                                     if any(common in symptom.lower() 
                                           for common in self.common_symptoms))
            
            # If all symptoms are common, prioritize common conditions
            all_symptoms_common = common_symptom_count == len(processed_symptoms)
            
            for idx, row in self.intents_df.iterrows():
                disease_score = 0.0
                matched_symptoms = set()
                
                # For each user symptom
                for user_symptom in processed_symptoms:
                    best_symptom_match = 0.0
                    
                    # Compare with each disease symptom
                    for col in self.symptom_columns:
                        disease_symptom = row[col]
                        if pd.isna(disease_symptom):
                            continue
                            
                        # Calculate similarity
                        similarity = self.calculate_symptom_similarity(user_symptom, disease_symptom)
                        
                        # Adjust similarity based on symptom commonality
                        if any(common in user_symptom.lower() for common in self.common_symptoms):
                            similarity *= 0.8  # Reduce weight for common symptoms
                        
                        # Apply column weight
                        weighted_similarity = similarity * self.symptom_weights[col]
                        
                        if weighted_similarity > best_symptom_match:
                            best_symptom_match = weighted_similarity
                    
                    disease_score += best_symptom_match
                
                # Normalize score by number of symptoms
                if processed_symptoms:
                    disease_score /= len(processed_symptoms)
                
                # Adjust score based on disease severity and symptom commonality
                disease_name = row['Disease']
                if all_symptoms_common:
                    # Boost score for common diseases if all symptoms are common
                    if disease_name in self.disease_severity['low']:
                        disease_score *= 1.5
                    elif disease_name in self.disease_severity['medium']:
                        disease_score *= 1.0
                    elif disease_name in self.disease_severity['high']:
                        disease_score *= 0.5
                else:
                    # If not all symptoms are common, maintain normal scoring
                    if disease_name in self.disease_severity['low']:
                        disease_score *= 1.0
                    elif disease_name in self.disease_severity['medium']:
                        disease_score *= 1.0
                    elif disease_name in self.disease_severity['high']:
                        disease_score *= 0.8
                
                disease_scores.append({
                    'index': idx,
                    'disease': disease_name,
                    'score': disease_score
                })
            
            # Sort diseases by score
            disease_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Adjust threshold based on symptom commonality
            base_threshold = 0.3
            if all_symptoms_common:
                threshold = base_threshold * 1.2  # Higher threshold for common symptoms
            else:
                threshold = base_threshold
            
            matching_indices = [d['index'] for d in disease_scores if d['score'] > threshold]
            
            if not matching_indices:
                return pd.DataFrame()
            
            matches = self.intents_df.loc[matching_indices].copy()
            matches['match_score'] = [d['score'] for d in disease_scores if d['score'] > threshold]
            
            print(f"Found {len(matches)} matches with scores above {threshold}:")
            for _, match in matches.iterrows():
                print(f"Disease: {match['Disease']}, Score: {match['match_score']:.2f}")
            
            return matches.sort_values('match_score', ascending=False)
            
        except Exception as e:
            print(f"Error in find_matching_diseases: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def generate_response(self, disease, language='en'):
        """Generate response with language support"""
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
            if language == 'en':
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
            else:
                # Hindi response templates
                confidence_hindi = {
                    'high': 'उच्च',
                    'moderate': 'मध्यम',
                    'low': 'कम'
                }[confidence_level]

                disease_name_hi = self.translate_text(disease['Disease'], 'en', 'hi')
                
                response = [
                    f"आपके लक्षणों के आधार पर, {confidence_hindi} संभावना है कि यह {disease_name_hi} हो सकता है।"
                ]
                
                if symptom_list:
                    translated_symptoms = self.translate_text(symptom_list, 'en', 'hi')
                    response.append(f"सामान्य लक्षणों में शामिल हैं: {translated_symptoms}")
                
                if medication_list:
                    translated_medications = self.translate_text(medication_list, 'en', 'hi')
                    response.append(f"संभावित उपचार में शामिल हो सकते हैं: {translated_medications}")
                
                if confidence_level != "high":
                    response.append("लक्षण मिलान स्तर को देखते हुए, कृपया अन्य संभावित स्थितियों पर भी विचार करें।")
                    
                response.append("कृपया ध्यान दें: यह एक निश्चित निदान नहीं है। उचित चिकित्सा सलाह के लिए डॉक्टर से संपर्क करें।")
            
            return ' '.join(response)
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            error_msg = "Sorry, I encountered an error. Please try again." if language == 'en' else "क्षमा करें, एक त्रुटि हुई। कृपया पुनः प्रयास करें।"
            return error_msg

    def detect_conversation_type(self, text, language='en'):
        """Detect if the input is a greeting or conversation starter"""
        if not text:
            return 'symptom'
            
        if language == 'hi':
            # Translate Hindi to English for pattern matching
            text = self.translate_text(text, 'hi', 'en')
        
        # Clean and normalize the text
        text = text.lower().strip()
        words = text.split()
        
        # First, check if it's a single word that matches any greeting
        if len(words) == 1:
            word = words[0]
            for conv_type, patterns in self.greeting_patterns.items():
                if word in patterns or any(pattern.startswith(word) for pattern in patterns):
                    return conv_type
        
        # Then check for exact matches
        for conv_type, patterns in self.greeting_patterns.items():
            if text in patterns:
                return conv_type
        
        # Check for pattern containment
        for conv_type, patterns in self.greeting_patterns.items():
            for pattern in patterns:
                # Check if the pattern is a phrase (multiple words)
                if ' ' in pattern:
                    if pattern in text:
                        return conv_type
                # For single word patterns, check if they appear as whole words
                else:
                    for word in words:
                        if pattern == word:
                            return conv_type
        
        # Check for combined greetings (e.g., "hi, how are you?")
        has_greeting = any(word in self.greeting_patterns['greetings'] for word in words)
        has_how_are_you = any(phrase in text for phrase in self.greeting_patterns['how_are_you'])
        if has_greeting and has_how_are_you:
            return 'how_are_you'
            
        # Check for questions about the chatbot's capabilities
        if any(word in ['what', 'how'] for word in words) and any(word in ['you', 'your'] for word in words):
            if any(word in ['do', 'can', 'could', 'help', 'assist'] for word in words):
                return 'help_requests'
        
        # If the text is very short (1-2 words) and contains a greeting word
        if len(words) <= 2:
            for word in words:
                for patterns in self.greeting_patterns.values():
                    if word in patterns:
                        return 'greetings'
        
        # Only if none of the above match, check if it might be a symptom
        medical_keywords = ['pain', 'hurt', 'feel', 'suffering', 'having', 'got', 'experiencing', 'symptom', 'problem', 'issue']
        if any(word in text for word in medical_keywords):
            return 'symptom'
        
        # For very short inputs that didn't match greetings, ask for clarification
        if len(words) <= 2:
            return 'help_requests'
        
        return 'symptom'  # Default to symptom if no other patterns match

    def get_conversation_response(self, conv_type, language='en'):
        """Get appropriate response for the conversation type"""
        if conv_type not in self.response_templates:
            return None
        
        # Get a random response for variety
        response = np.random.choice(self.response_templates[conv_type])
        
        # Add follow-up prompt for symptoms if not a goodbye
        if conv_type != 'goodbyes' and 'symptoms' not in response.lower():
            response += " What symptoms are you experiencing?"
        
        # Translate if needed
        if language == 'hi':
            response = self.translate_text(response, 'en', 'hi')
        
        return response

    def chat(self):
        print("Healthcare Chatbot: Hello! How can I assist you today?")
        self.text_to_speech("Hello! How can I assist you today?")
        
        symptoms = []
        conversation_state = 'greeting'  # Track conversation state
        
        while True:
            user_input = input("You: ")
            
            # Detect conversation type
            conv_type = self.detect_conversation_type(user_input)
            
            # Handle conversation patterns
            if conv_type != 'symptom':
                response = self.get_conversation_response(conv_type)
                print(f"Healthcare Chatbot: {response}")
                self.text_to_speech(response)
                
                # If it's a goodbye, end the conversation
                if conv_type == 'goodbyes':
                    break
                    
                # Update conversation state
                conversation_state = conv_type
                continue
            
            # If we reach here, treat input as symptom
            if user_input.lower() in ['no', 'nope', 'done', 'that\'s all', 'thats all', 'nothing else']:
                break
            
            # If we were in greeting state, provide more guidance
            if conversation_state == 'greeting':
                print("Healthcare Chatbot: I'm here to help identify possible health conditions. Please describe your symptoms.")
                self.text_to_speech("I'm here to help identify possible health conditions. Please describe your symptoms.")
                conversation_state = 'symptoms'
                continue
            
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