import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from spellchecker import SpellChecker
from translate import Translator
import json
import re
from datetime import datetime

class EnhancedNLP:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Initialize components
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.spell_checker = SpellChecker()
        self.medical_terms = self.load_medical_terms()
        self.conversation_history = []
        self.user_preferences = {}
        
    def load_medical_terms(self):
        """Load medical terms and their synonyms"""
        medical_terms = {
            "fever": ["pyrexia", "hyperthermia", "temperature", "febrile"],
            "headache": ["cephalgia", "migraine", "head pain"],
            "cough": ["tussis", "coughing", "hack"],
            "fatigue": ["tiredness", "exhaustion", "lethargy", "weakness"],
            "nausea": ["sickness", "queasiness", "upset stomach"],
            "pain": ["ache", "discomfort", "soreness", "distress"],
            # Add more medical terms and their synonyms
        }
        return medical_terms
    
    def analyze_sentiment(self, text):
        """Analyze the emotional state of the user's input"""
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Determine emotional state
        if sentiment_scores['compound'] <= -0.5:
            emotional_state = "very_distressed"
        elif sentiment_scores['compound'] <= -0.1:
            emotional_state = "concerned"
        elif sentiment_scores['compound'] >= 0.5:
            emotional_state = "positive"
        elif sentiment_scores['compound'] >= 0.1:
            emotional_state = "neutral_positive"
        else:
            emotional_state = "neutral"
            
        return {
            'emotional_state': emotional_state,
            'sentiment_scores': sentiment_scores
        }
    
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
    
    def standardize_medical_terms(self, text):
        """Convert medical terms to their standard form"""
        words = text.lower().split()
        standardized_words = []
        
        for word in words:
            standardized = word
            for standard_term, synonyms in self.medical_terms.items():
                if word in [syn.lower() for syn in synonyms]:
                    standardized = standard_term
                    break
            standardized_words.append(standardized)
        
        return ' '.join(standardized_words)
    
    def translate_text(self, text, target_lang='en'):
        """Translate text to/from English"""
        try:
            translator = Translator(to_lang=target_lang)
            translated = translator.translate(text)
            return translated
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text
    
    def add_to_history(self, user_input, bot_response, context=None):
        """Add interaction to conversation history"""
        timestamp = datetime.now().isoformat()
        
        # Analyze sentiment of user input
        sentiment = self.analyze_sentiment(user_input)
        
        interaction = {
            'timestamp': timestamp,
            'user_input': user_input,
            'bot_response': bot_response,
            'sentiment': sentiment,
            'context': context or {}
        }
        
        self.conversation_history.append(interaction)
        
        # Maintain only last 10 interactions to manage memory
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
    
    def get_follow_up_questions(self, current_symptoms):
        """Generate relevant follow-up questions based on current symptoms"""
        follow_up_questions = []
        
        # Common follow-up patterns
        symptom_questions = {
            "fever": [
                "How long have you had the fever?",
                "Have you measured your temperature?",
                "Are you experiencing chills or sweating?"
            ],
            "headache": [
                "Is the headache concentrated in any particular area?",
                "Does light or sound make it worse?",
                "Have you had any recent head injuries?"
            ],
            "cough": [
                "Is it a dry cough or productive cough?",
                "How long have you been coughing?",
                "Are you coughing up anything?"
            ],
            # Add more symptom-specific questions
        }
        
        # Add relevant follow-up questions based on current symptoms
        for symptom in current_symptoms:
            if symptom.lower() in symptom_questions:
                follow_up_questions.extend(symptom_questions[symptom.lower()])
        
        return follow_up_questions
    
    def update_user_preferences(self, user_id, preferences):
        """Update user preferences"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id].update(preferences)
    
    def get_user_preferences(self, user_id):
        """Get user preferences"""
        return self.user_preferences.get(user_id, {})
    
    def get_conversation_context(self):
        """Get relevant context from conversation history"""
        if not self.conversation_history:
            return {}
        
        # Get the last 3 interactions
        recent_interactions = self.conversation_history[-3:]
        
        # Extract symptoms mentioned
        mentioned_symptoms = set()
        emotional_states = []
        
        for interaction in recent_interactions:
            # Get symptoms from user input
            user_input = interaction['user_input'].lower()
            for term in self.medical_terms:
                if term in user_input or any(syn.lower() in user_input for syn in self.medical_terms[term]):
                    mentioned_symptoms.add(term)
            
            # Track emotional states
            emotional_states.append(interaction['sentiment']['emotional_state'])
        
        return {
            'recent_symptoms': list(mentioned_symptoms),
            'emotional_states': emotional_states,
            'interaction_count': len(self.conversation_history)
        } 