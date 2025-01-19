import speech_recognition as sr
import pyttsx3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import pandas as pd
import re
from googletrans import Translator
from langdetect import detect

class MultilingualHealthcareChatbot:
    def __init__(self, default_language='en'):
        # Speech and text processing
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        
        # Multilingual support
        self.translator = Translator()
        self.default_language = default_language
        
        # Language-specific preprocessing
        self.language_configs = {
            'en': {
                'lemmatizer': WordNetLemmatizer(),
                'stop_words': set(stopwords.words('english')),
                'stemmer': SnowballStemmer('english')
            },
            'hi': {
                'lemmatizer': None,  # Hindi lemmatization is complex
                'stop_words': set(),  # Hindi stop words
                'stemmer': None  # Remove Hindi stemming
            }
        }
        
        # Load intents dataset
        self.intents_df = self.load_intents('MedDatasetFinal_modeled.csv')
        self.symptom_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']

    def detect_language(self, text):
        """Detect language of the input text"""
        try:
            return detect(text)
        except:
            return self.default_language

    def translate_text(self, text, target_language='en'):
        """Translate text between languages"""
        try:
            translation = self.translator.translate(text, dest=target_language)
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def speech_to_text(self, language=None):
        """
        Convert speech to text with language-specific recognition
        """
        language = language or self.default_language
        recognizer = sr.Recognizer()
        
        try:
            with sr.Microphone() as source:
                print(f"\nListening ({language})...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=5)
                
                text = recognizer.recognize_google(audio, language='en-US' if language == 'en' else 'hi-IN')
                print(f"You said: {text}")
                return text
        except sr.RequestError:
            print("Could not request results; check your internet connection")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def text_to_speech(self, text, language=None):
        """
        Convert text to speech with language support
        """
        try:
            engine = pyttsx3.init()
            if language == 'hi':
                engine.setProperty('rate', 145)  # Slower rate for Hindi
            else:
                engine.setProperty('rate', 175)  # Normal rate for English
                
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Text-to-speech error: {str(e)}")

    def preprocess_text(self, text, language=None):
        """
        Language-aware text preprocessing
        """
        language = language or self.detect_language(text)
        config = self.language_configs.get(language, self.language_configs['en'])
        
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize/stem
        if config['lemmatizer']:
            tokens = [config['lemmatizer'].lemmatize(token) for token in tokens 
                      if token not in config['stop_words']]
        else:
            # For languages without lemmatization, use stemming
            tokens = [config['stemmer'].stem(token) for token in tokens 
                      if token not in config['stop_words']]
        
        return ' '.join(tokens)

    def chat(self):
        """
        Multilingual chatbot conversation flow
        """
        # Language selection
        print("\nSelect your language / अपनी भाषा चुनें:")
        print("E - English")
        print("H - हिंदी (Hindi)")
        
        while True:
            lang_choice = input("Enter E or H: ").upper()
            if lang_choice in ['E', 'H']:
                break
            print("Invalid choice. Please enter E or H.")
        
        # Set language based on choice
        language = 'hi' if lang_choice == 'H' else 'en'
        
        # Welcome message in selected language
        welcome_messages = {
            'en': "Hello! How can I assist you today? Please describe your main symptom.",
            'hi': "नमस्ते! आज मैं आपकी कैसे सहायता कर सकता हूँ? कृपया अपने मुख्य लक्षण का वर्णन करें।"
        }
        
        print(f"\nHealthcare Chatbot: {welcome_messages[language]}")
        self.text_to_speech(welcome_messages[language], language)
        
        symptoms = []
        while True:
            # Capture symptoms in user's language
            user_input = self.speech_to_text(language)
            if user_input is None:
                continue
            
            # Translate to English for processing if needed
            if language != 'en':
                user_input = self.translate_text(user_input, 'en')
            
            preprocessed_input = self.preprocess_text(user_input, 'en')
            symptoms.append(preprocessed_input)
            
            # Prompt for more symptoms
            more_symptom_messages = {
                'en': "Any other symptoms? (Say 'no' if you're finished listing symptoms)",
                'hi': "क्या आपके पास कोई अन्य लक्षण हैं? (समाप्त करने के लिए 'नहीं' कहें)"
            }
            
            print(f"Healthcare Chatbot: {more_symptom_messages[language]}")
            self.text_to_speech(more_symptom_messages[language], language)
            
            user_input = self.speech_to_text(language)
            if user_input and user_input.lower() in ['no', 'नहीं']:
                break
        
        if not symptoms:
            no_symptoms_messages = {
                'en': "No symptoms were provided. I can't make a prediction without any symptoms.",
                'hi': "कोई लक्षण नहीं दिए गए। बिना किसी लक्षण के मैं भविष्यवाणी नहीं कर सकता।"
            }
            print(f"Healthcare Chatbot: {no_symptoms_messages[language]}")
            self.text_to_speech(no_symptoms_messages[language], language)
            return

        # Find matching diseases
        matching_diseases = self.find_matching_diseases(symptoms)
        
        if matching_diseases.empty:
            no_match_messages = {
                'en': "I couldn't find any matching conditions for your symptoms. Please consult a healthcare professional for proper diagnosis.",
                'hi': "मुझे आपके लक्षणों से मेल खाने वाली कोई स्थिति नहीं मिली। कृपया उचित निदान के लिए एक स्वास्थ्य पेशेवर से परामर्श करें।"
            }
            response = no_match_messages[language]
        else:
            # Select the disease that matches the most symptoms
            disease_match_count = matching_diseases[self.symptom_columns].apply(
                lambda x: x.str.contains('|'.join(symptoms), case=False, na=False).sum(), 
                axis=1
            )
            best_match = matching_diseases.loc[disease_match_count.idxmax()]
            
            # Generate response
            response = self.generate_response(best_match)
            
            # Translate response back to user's language if needed
            if language != 'en':
                response = self.translate_text(response, language)
        
        print(f"Healthcare Chatbot: {response}")
        self.text_to_speech(response, language)

    # Rest of the methods remain the same as in the original implementation
    def load_intents(self, file_path):
        # Same as original method
        df = pd.read_csv(file_path)
        print("Columns in the CSV file:", df.columns)
        print("First few rows of the dataframe:")
        print(df.head())
        return df

    def find_matching_diseases(self, symptoms):
        """Find diseases matching the given symptoms"""
        # Create a mask for matching symptoms
        mask = pd.DataFrame(False, index=self.intents_df.index, columns=[0])
        
        for symptom in symptoms:
            symptom_match = self.intents_df[self.symptom_columns].apply(
                lambda x: x.str.contains(symptom, case=False, na=False)
            ).any(axis=1)
            mask = mask | symptom_match.to_frame()
        
        return self.intents_df[mask[0]]

    def generate_response(self, disease):
        # Same as original method
        symptom_list = ', '.join([symptom for symptom in disease[self.symptom_columns] if pd.notna(symptom) and symptom.lower() != 'unknown'])
        medication_columns = ['Medication_1', 'Medication_2', 'Medication_3', 'Medication_4']
        medication_list = ', '.join([medication for medication in disease[medication_columns] if pd.notna(medication) and medication.lower() != 'unknown'])
        
        response = f"Based on your symptoms, it could be {disease['Disease']}. Common symptoms include {symptom_list}. "
        if medication_list:
            response += f"Typical treatments might involve {medication_list}. "
        response += "Please consult a healthcare professional for proper diagnosis and treatment."
        return response

if __name__ == "__main__":
    try:
        chatbot = MultilingualHealthcareChatbot()
        chatbot.chat()
    except Exception as e:
        print(f"An error occurred: {str(e)}")