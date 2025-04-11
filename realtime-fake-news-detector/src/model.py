from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, Union, Tuple
import joblib
import logging
from .preprocessor import TextPreprocessor

class FakeNewsDetector:
    def __init__(self):
        """Initialize the fake news detector with necessary components."""
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = PassiveAggressiveClassifier(max_iter=1000)
        self.logger = logging.getLogger(__name__)
        
        # Initialize with some dummy data for immediate testing
        self._initialize_dummy_model()

    def _initialize_dummy_model(self):
        """Initialize a basic model with dummy data for testing."""
        try:
            # Dummy dataset (replace with real dataset in production)
            dummy_texts = [
                "Breaking news: Shocking revelation about government conspiracy",
                "Scientists discover new species in Amazon rainforest",
                "Click here to win million dollar lottery guaranteed",
                "Local community hosts annual charity event",
                "Miracle cure for all diseases found in common fruit"
            ]
            dummy_labels = [1, 0, 1, 0, 1]  # 1 for fake, 0 for real

            # Fit vectorizer and classifier
            X = self.vectorizer.fit_transform(dummy_texts)
            self.classifier.fit(X, dummy_labels)
            
        except Exception as e:
            self.logger.error(f"Error initializing dummy model: {str(e)}")

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict whether a news article is fake or real.
        
        Args:
            text (str): Input news text
            
        Returns:
            Dict: Prediction results including label and confidence score
        """
        try:
            # Preprocess the text
            preprocessed_text = self.preprocessor.preprocess_text(text)
            
            # Vectorize the text
            text_vector = self.vectorizer.transform([preprocessed_text])
            
            # Get prediction and confidence score
            prediction = self.classifier.predict(text_vector)[0]
            confidence_score = abs(self.classifier.decision_function(text_vector)[0])
            
            # Normalize confidence score to 0-1 range
            confidence_score = min(confidence_score / 2.0, 1.0)
            
            # Prepare result
            result = {
                'label': 'FAKE' if prediction == 1 else 'REAL',
                'confidence': round(confidence_score * 100, 2),
                'features': self._extract_key_features(preprocessed_text)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return {
                'label': 'ERROR',
                'confidence': 0.0,
                'features': []
            }

    def _extract_key_features(self, text: str, top_n: int = 5) -> list:
        """
        Extract key features that influenced the prediction.
        
        Args:
            text (str): Preprocessed text
            top_n (int): Number of top features to return
            
        Returns:
            list: List of key features
        """
        try:
            # Get feature names and their TF-IDF scores
            text_vector = self.vectorizer.transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get non-zero features
            non_zero = text_vector.nonzero()[1]
            scores = text_vector.data
            
            # Sort features by score
            feature_scores = [(feature_names[i], scores[idx]) 
                            for idx, i in enumerate(non_zero)]
            feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
            
            # Return top N features
            return feature_scores[:top_n]
            
        except Exception as e:
            self.logger.error(f"Error extracting key features: {str(e)}")
            return []

    def update_model(self, text: str, label: int) -> bool:
        """
        Update the model with new labeled data.
        
        Args:
            text (str): News text
            label (int): True label (0 for real, 1 for fake)
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Preprocess and vectorize the text
            preprocessed_text = self.preprocessor.preprocess_text(text)
            text_vector = self.vectorizer.transform([preprocessed_text])
            
            # Update the classifier
            self.classifier.partial_fit(text_vector, [label], classes=[0, 1])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")
            return False

    def analyze_text_characteristics(self, text: str) -> Dict:
        """
        Analyze various characteristics of the input text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Various text characteristics
        """
        try:
            # Basic text statistics
            word_count = len(text.split())
            sentence_count = len(text.split('.'))
            avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
            
            # Calculate more sophisticated metrics
            clickbait_words = ['shocking', 'amazing', 'incredible', 'won\'t believe', 'miracle']
            clickbait_count = sum(1 for word in clickbait_words if word in text.lower())
            
            return {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': round(avg_word_length, 2),
                'clickbait_indicators': clickbait_count
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing text characteristics: {str(e)}")
            return {}

    def save_model(self, path: str) -> bool:
        """
        Save the current model state to disk.
        
        Args:
            path (str): Path to save the model
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            model_state = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }
            joblib.dump(model_state, path)
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, path: str) -> bool:
        """
        Load a saved model state from disk.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            bool: True if load successful, False otherwise
        """
        try:
            model_state = joblib.load(path)
            self.vectorizer = model_state['vectorizer']
            self.classifier = model_state['classifier']
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
