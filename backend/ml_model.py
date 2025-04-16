import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
from fuzzywuzzy import fuzz
import re
from sklearn.impute import SimpleImputer
import json
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

class DiseasePredictor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.symptom_combinations = []
        self.symptom_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5']
        self.best_params = None
        self.all_symptoms = set()
        self.group_to_diseases = {}  # Initialize group mapping
        
        # Get the directory path where ml_model.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_file = os.path.join(current_dir, 'disease_prevalence.json')
        
        # Load prevalence information
        try:
            with open(json_file, 'r') as f:
                self.prevalence_config = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find {json_file}")
            # Provide default configuration
            self.prevalence_config = {
                "prevalence_scores": {},
                "severity_weights": {
                    "Low": 1.0,
                    "Medium": 1.5,
                    "High": 2.0,
                    "Severe": 2.5
                },
                "symptom_weights": {
                    "Symptom_1": 1.5,
                    "Symptom_2": 1.3,
                    "Symptom_3": 1.2,
                    "Symptom_4": 1.1,
                    "Symptom_5": 1.0
                }
            }
        
        # Force new training
        print("Training new model...")
        self.train_model()
        
    def preprocess_text(self, text):
        """Preprocess text by converting to lowercase and removing special characters and common phrases"""
        if not isinstance(text, str):
            return str(text).lower()
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove common phrases and extra words
        phrases_to_remove = [
            'i have', 'i feel', 'i feel like', 'i am experiencing',
            'i got', 'i\'ve got', 'i\'m having', 'i\'m experiencing',
            'i am having', 'i am feeling', 'i am suffering from',
            'i suffer from', 'i am', 'i\'m', 'like', 'a', 'an', 'the',
            'some', 'any', 'my', 'me', 'myself'
        ]
        
        for phrase in phrases_to_remove:
            text = text.replace(phrase, '').strip()
        
        # Remove extra spaces and special characters
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        
        # Handle common symptoms and their variations
        variations = {
            'fever': ['fever', 'feverish', 'temperature', 'high temperature', 'hot', 'burning sensation', 'high fever'],
            'headache': ['headache', 'head pain', 'head ache', 'migraine', 'hedche', 'hedache', 'headach', 'head pressure', 'head hurts'],
            'cough': ['cough', 'coughing', 'coughs', 'dry cough', 'wet cough', 'persistent cough'],
            'cold': ['cold', 'runny nose', 'stuffy nose', 'nasal congestion', 'sneezing', 'common cold'],
            'sore throat': ['sore throat', 'throat pain', 'throat ache', 'painful throat', 'throat irritation'],
            'body ache': ['body ache', 'muscle pain', 'joint pain', 'body pain', 'aches', 'pains', 'muscular pain', 'joint pin', 'leg pin', 'leg pain', 'limb pain', 'joint ache', 'leg ache'],
            'fatigue': ['fatigue', 'tired', 'tiredness', 'exhaustion', 'weakness', 'lethargy', 'feeling weak'],
            'nausea': ['nausea', 'nauseous', 'queasy', 'sick to stomach', 'wanting to vomit', 'feeling sick'],
            'vomiting': ['vomiting', 'vomit', 'throwing up', 'puking', 'vomitted', 'vommiting'],
            'diarrhea': ['diarrhea', 'loose stools', 'watery stools', 'frequent bowel movements'],
            'chest pain': ['chest pain', 'chest tightness', 'chest pressure', 'chest discomfort'],
            'breathing difficulty': ['breathing difficulty', 'shortness of breath', 'breathlessness', 'dyspnea'],
            'stomach pain': ['stomach pain', 'abdominal pain', 'belly pain', 'tummy ache', 'stomach ache'],
            'dizziness': ['dizziness', 'dizzy', 'lightheaded', 'vertigo', 'spinning sensation'],
            'weakness': ['weakness', 'weak', 'lack of energy', 'loss of strength', 'feeling weak']
        }
        
        # Replace variations with standard terms
        for standard, variants in variations.items():
            if any(variant in text for variant in variants):
                return standard
        
        return text

    def preprocess_data(self):
        """Preprocess the dataset with improved feature engineering and prevalence weighting"""
        print("\n=== Starting Data Preprocessing ===")
        print(f"Loading dataset from: {self.dataset_path}")
        
        try:
            # Read the dataset with explicit encoding and handle potential issues
            df = pd.read_csv(self.dataset_path, encoding='utf-8')
            
            # Print column names for debugging
            print("Dataset columns:", df.columns.tolist())
            
            # Verify required columns exist
            required_columns = ['Disease_Name', 'Category', 'Severity_Level', 
                              'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            
            # Group similar diseases based on category and symptoms
            print("\nGrouping similar diseases...")
            disease_groups = {}
            for _, row in df.iterrows():
                disease = row['Disease_Name']
                category = row['Category']
                symptoms = set(row[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5']].dropna())
                
                # Create a key based on category and main symptoms
                key = (category, tuple(sorted(symptoms)[:3]))  # Use first 3 symptoms for grouping
                
                if key not in disease_groups:
                    disease_groups[key] = []
                disease_groups[key].append(disease)
            
            # Create a mapping from disease to group
            disease_to_group = {}
            self.group_to_diseases = {}  # Reset group mapping
            for group_id, (key, diseases) in enumerate(disease_groups.items()):
                group_name = f"Group_{group_id}"
                self.group_to_diseases[group_name] = diseases  # Store group mapping
                for disease in diseases:
                    disease_to_group[disease] = group_name
            
            # Add group information to the dataset
            df['Disease_Group'] = df['Disease_Name'].map(disease_to_group)
            
            # Add prevalence scores
            df['Prevalence_Score'] = df['Disease_Name'].map(
                lambda x: self.prevalence_config['prevalence_scores'].get(x, 1)
            )
            
            # Add severity weights
            df['Severity_Weight'] = df['Severity_Level'].map(
                self.prevalence_config['severity_weights']
            )
            
            # Calculate final weight combining prevalence and severity
            df['Weight'] = df['Prevalence_Score'] * df['Severity_Weight']
            
            # Get all unique symptoms and preprocess them
            self.all_symptoms = set()
            for col in self.symptom_columns:
                symptoms = df[col].dropna().unique()
                for symptom in symptoms:
                    processed_symptom = self.preprocess_text(symptom)
                    self.all_symptoms.add(processed_symptom)
            
            # Create binary features for symptoms with weights
            symptom_features = pd.DataFrame(0, index=df.index, columns=sorted(self.all_symptoms))
            
            for col in self.symptom_columns:
                weight = self.prevalence_config['symptom_weights'][col]
                for symptom in df[col].dropna():
                    processed_symptom = self.preprocess_text(symptom)
                    if processed_symptom in symptom_features.columns:
                        symptom_features[processed_symptom] = symptom_features[processed_symptom] + weight * (df[col] == symptom)
            
            # Add symptom count feature
            df['symptom_count'] = df[self.symptom_columns].notna().sum(axis=1)
            
            # Create symptom combinations with weights
            symptom_combinations = []
            for i in range(len(self.symptom_columns)-1):
                for j in range(i+1, len(self.symptom_columns)):
                    col1, col2 = self.symptom_columns[i], self.symptom_columns[j]
                    weight = (self.prevalence_config['symptom_weights'][col1] + 
                            self.prevalence_config['symptom_weights'][col2]) / 2
                    valid_pairs = df[[col1, col2]].dropna()
                    for _, row in valid_pairs.iterrows():
                        if pd.notna(row[col1]) and pd.notna(row[col2]):
                            symptom1 = self.preprocess_text(row[col1])
                            symptom2 = self.preprocess_text(row[col2])
                            combo = f"{symptom1}_{symptom2}"
                            if combo not in symptom_combinations:
                                symptom_combinations.append(combo)
            
            # Add combination features
            combination_features = pd.DataFrame(0, index=df.index, columns=symptom_combinations)
            for combo in symptom_combinations:
                symptoms = combo.split('_')
                if len(symptoms) == 2:
                    symptom1, symptom2 = symptoms
                    combination_features[combo] = (symptom_features[symptom1] > 0) & (symptom_features[symptom2] > 0)
            
            # Encode categorical variables
            categorical_features = ['Category', 'Severity_Level']
            for feature in categorical_features:
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(df[feature])
            
            # Combine all features
            self.feature_names = sorted(self.all_symptoms)
            self.symptom_combinations = symptom_combinations
            
            # Create final feature set
            X = pd.concat([
                symptom_features,
                combination_features,
                df[['symptom_count', 'Weight'] + categorical_features]
            ], axis=1)
            
            # Encode target variable (using disease groups instead of individual diseases)
            self.label_encoders['Disease_Group'] = LabelEncoder()
            y = self.label_encoders['Disease_Group'].fit_transform(df['Disease_Group'])
            
            print(f"\nReduced number of classes from {len(df['Disease_Name'].unique())} to {len(np.unique(y))}")
            print("\n=== Data Preprocessing Completed ===")
            return X, y, df
            
        except Exception as e:
            print(f"\nError during preprocessing: {str(e)}")
            print("\nFull error details:")
            import traceback
            traceback.print_exc()
            raise
    
    def train_model(self):
        """Train the model with improved parameters and prevalence weighting"""
        print("Starting model training...")
        
        # Preprocess data
        X, y, df = self.preprocess_data()
        
        # Handle NaN values
        X = X.fillna(0)
        
        # Analyze class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print("\nClass distribution before handling:")
        for cls, count in zip(unique_classes, class_counts):
            disease_name = self.label_encoders['Disease_Group'].inverse_transform([cls])[0]
            print(f"{disease_name}: {count} samples")
        
        # Remove classes with only 1 sample and merge rare classes
        min_samples = 3  # Minimum samples per class
        rare_classes = unique_classes[class_counts < min_samples]
        
        if len(rare_classes) > 0:
            print("\nMerging rare classes:")
            for cls in rare_classes:
                disease_name = self.label_encoders['Disease_Group'].inverse_transform([cls])[0]
                print(f"Merging {disease_name} (only {class_counts[np.where(unique_classes == cls)[0][0]]} samples)")
            
            # Create a new label encoder for merged classes
            merged_label_encoder = LabelEncoder()
            
            # Create a mapping for rare classes to merge them into "Other" category
            class_mapping = {cls: 'Other' for cls in rare_classes}
            y_str = np.array([class_mapping.get(cls, self.label_encoders['Disease_Group'].inverse_transform([cls])[0]) for cls in y])
            
            # Fit the new label encoder
            y = merged_label_encoder.fit_transform(y_str)
            
            # Store the merged label encoder
            self.merged_label_encoder = merged_label_encoder
            
            # Update unique classes after merging
            unique_classes = np.unique(y)
            print(f"\nNumber of classes after merging rare ones: {len(unique_classes)}")
            print("New class distribution:")
            for cls, count in zip(unique_classes, np.bincount(y)):
                class_name = merged_label_encoder.inverse_transform([cls])[0]
                print(f"{class_name}: {count} samples")
        
        # Apply SMOTE for common diseases only if we have enough samples
        common_diseases = [disease for disease, score in self.prevalence_config['prevalence_scores'].items() if score >= 4]
        common_indices = [i for i, disease in enumerate(y) if self.label_encoders['Disease_Group'].inverse_transform([disease])[0] in common_diseases]
        
        if len(common_indices) > 5:  # Only apply SMOTE if we have at least 5 samples
            X_common = X.iloc[common_indices]
            y_common = y[common_indices]
            
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, len(common_indices)-1))
                X_resampled, y_resampled = smote.fit_resample(X_common, y_common)
                
                # Combine resampled data with original data
                X = pd.concat([X, X_resampled])
                y = np.concatenate([y, y_resampled])
                print(f"Applied SMOTE to {len(common_indices)} common disease samples")
            except Exception as e:
                print(f"Could not apply SMOTE: {str(e)}")
                print("Continuing with original dataset")
        
        # Use a fixed test size of 0.2 but ensure minimum samples per class
        test_size = 0.2
        min_samples_per_class = 2
        
        # Calculate if we have enough samples for stratification
        can_stratify = all(np.bincount(y) >= min_samples_per_class)
        
        if can_stratify:
            print("\nUsing stratified split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=42, 
                stratify=y
            )
        else:
            print("\nNot enough samples for stratification, using simple split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=42
            )
        
        # Define parameter grids for different models
        param_grids = {
            'gradient_boosting': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__subsample': [0.8, 0.9],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            }
        }
        
        # Create pipeline with SimpleImputer to handle NaN values
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
        
        # Use simple train-test split for model training
        print("\nTraining model with simple train-test split...")
        pipeline.fit(X_train, y_train)
        self.model = pipeline

        # Calculate and display both training and testing accuracy
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"\nTraining accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}")
        
        # Generate detailed classification report
        y_pred_test = self.model.predict(X_test)
        print("\nDetailed Classification Report on Test Set:")
        print(classification_report(y_test, y_pred_test))

        # Save the model
        self.save_model()
        print("\nModel training completed")
        
    def predict_disease(self, symptoms):
        """Predict disease based on symptoms with prevalence weighting"""
        try:
            # Preprocess symptoms
            processed_symptoms = [self.preprocess_text(s) for s in symptoms]
            print(f"\nProcessing symptoms: {processed_symptoms}")
            
            # Create input features
            all_features = self.feature_names + self.symptom_combinations + ['symptom_count', 'Weight', 'Category', 'Severity_Level']
            input_features = pd.DataFrame(0, index=[0], columns=all_features)
            
            # Match symptoms using fuzzy matching
            matched_symptoms = []
            for symptom in processed_symptoms:
                best_match = None
                best_score = 0
                for known_symptom in self.all_symptoms:
                    score = fuzz.ratio(symptom.lower(), known_symptom.lower())
                    if score > best_score and score > 70:
                        best_score = score
                        best_match = known_symptom
                if best_match:
                    matched_symptoms.append(best_match)
                    print(f"Matched '{symptom}' to '{best_match}' with score {best_score}")
            
            if not matched_symptoms:
                print("No matching symptoms found in the database")
                return []
            
            print(f"\nMatched symptoms: {matched_symptoms}")
            
            # Set features for matched symptoms
            for symptom in matched_symptoms:
                if symptom in input_features.columns:
                    input_features[symptom] = 1
            
            # Add symptom count
            input_features['symptom_count'] = len(matched_symptoms)
            
            # Add default values for other features
            input_features['Weight'] = 1.0  # Default weight
            input_features['Category'] = 0  # Default category
            input_features['Severity_Level'] = 0  # Default severity
            
            # Common symptom combinations and their likely diseases
            # common_combinations = {
            #     frozenset(['fever', 'headache']): {
            #         'Common Cold': 0.8,
            #         'Influenza (Flu)': 0.7,
            #         'Viral Fever': 0.6,
            #         'Sinusitis': 0.5
            #     },
            #     frozenset(['fever']): {
            #         'Common Cold': 0.6,
            #         'Influenza (Flu)': 0.5,
            #         'Viral Fever': 0.5
            #     },
            #     frozenset(['headache']): {
            #         'Migraine': 0.6,
            #         'Tension Headache': 0.5,
            #         'Sinusitis': 0.4
            #     }
            # }
            
            # Make predictions
            try:
                probabilities = self.model.predict_proba(input_features)[0]
                group_indices = np.argsort(probabilities)[::-1]
                
                # Get top predictions with adjusted probabilities based on prevalence
                predictions = []
                current_symptoms = frozenset(matched_symptoms)
                
                # First, check for common symptom combinations
                # for combo, likely_diseases in common_combinations.items():
                #     if combo.issubset(current_symptoms):
                #         for disease, base_prob in likely_diseases.items():
                #             # Get the prevalence score for the disease
                #             prevalence_score = self.prevalence_config['prevalence_scores'].get(disease, 1)
                            
                #             # Adjust probability based on prevalence
                #             final_prob = base_prob * (1 + (prevalence_score - 1) * 0.2)
                            
                #             predictions.append({
                #                 'disease': disease,
                #                 'probability': final_prob,
                #                 'prevalence_score': prevalence_score
                #             })
                
                # Then add model predictions
                for idx in group_indices:
                    try:
                        group = self.label_encoders['Disease_Group'].inverse_transform([idx])[0]
                        if group not in self.group_to_diseases:
                            continue
                            
                        diseases_in_group = self.group_to_diseases[group]
                        group_prob = probabilities[idx]
                        
                        # Calculate average prevalence score for the group
                        group_prevalence = np.mean([
                            self.prevalence_config['prevalence_scores'].get(disease, 1)
                            for disease in diseases_in_group
                        ])
                        
                        # Only include predictions for diseases with sufficient prevalence
                        for disease in diseases_in_group:
                            disease_prevalence = self.prevalence_config['prevalence_scores'].get(disease, 1)
                            
                            # Skip rare diseases unless they have very high probability
                            if disease_prevalence < 2 and group_prob < 0.8:
                                continue
                                
                            # Adjust probability based on prevalence and group probability
                            final_prob = group_prob * (1 + (disease_prevalence - 1) * 0.2)
                            
                            # Only add if not already in predictions
                            if not any(p['disease'] == disease for p in predictions):
                                predictions.append({
                                    'disease': disease,
                                    'probability': final_prob,
                                    'prevalence_score': disease_prevalence
                                })
                    except Exception as e:
                        print(f"Warning: Error processing group {idx}: {str(e)}")
                        continue
                
                # Sort predictions by adjusted probability
                predictions.sort(key=lambda x: x['probability'], reverse=True)
                
                if not predictions:
                    print("No valid predictions found after processing groups")
                    return []
                    
                print(f"\nFound {len(predictions)} valid predictions")
                return predictions[:5]  # Return top 5 predictions
                
            except Exception as e:
                print(f"Error making prediction: {str(e)}")
                return []
            
        except Exception as e:
            print(f"Error in predict_disease: {str(e)}")
            return []
    
    def save_model(self):
        """Save the trained model and encoders"""
        model_dir = 'models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        joblib.dump(self.model, os.path.join(model_dir, 'disease_predictor.pkl'))
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
        joblib.dump(self.feature_names, os.path.join(model_dir, 'feature_names.pkl'))
        joblib.dump(self.symptom_combinations, os.path.join(model_dir, 'symptom_combinations.pkl'))
        if self.best_params:
            joblib.dump(self.best_params, os.path.join(model_dir, 'best_params.pkl'))
    
    def load_model(self):
        """Load the trained model and encoders"""
        model_dir = 'models'
        if not os.path.exists(model_dir):
            raise Exception("Model directory not found")
        
        self.model = joblib.load(os.path.join(model_dir, 'disease_predictor.pkl'))
        self.label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
        self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        self.symptom_combinations = joblib.load(os.path.join(model_dir, 'symptom_combinations.pkl'))
        if os.path.exists(os.path.join(model_dir, 'best_params.pkl')):
            self.best_params = joblib.load(os.path.join(model_dir, 'best_params.pkl'))

if __name__ == "__main__":
    print("\n=== Starting Disease Predictor Training ===")
    try:
        # Example usage
        print("\nInitializing DiseasePredictor...")
        predictor = DiseasePredictor('Priority_wise_MedicalDataset - Sheet1 (1).csv')
        
        # Test prediction
        print("\nTesting prediction with sample symptoms...")
        test_symptoms = ['Fever', 'Headache']
        predictions = predictor.predict_disease(test_symptoms)
        
        print("\nTop 3 predicted diseases:")
        for pred in predictions:
            print(f"\nDisease: {pred['disease']}")
            print(f"Probability: {pred['probability']:.2f}")
            print(f"Prevalence Score: {pred['prevalence_score']}")
            
        print("\n=== Training and Prediction Completed ===")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc() 