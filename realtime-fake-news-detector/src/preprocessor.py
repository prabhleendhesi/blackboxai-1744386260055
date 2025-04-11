import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List
import logging

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
except Exception as e:
    logging.error(f"Failed to download NLTK data: {str(e)}")

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with necessary tools."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing special characters and normalizing.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            self.logger.error(f"Error in clean_text: {str(e)}")
            return text

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize the input text into words.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        try:
            return word_tokenize(text)
        except Exception as e:
            self.logger.error(f"Error in tokenize_text: {str(e)}")
            return []

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from the token list.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: List of tokens with stopwords removed
        """
        try:
            return [token for token in tokens if token not in self.stop_words]
        except Exception as e:
            self.logger.error(f"Error in remove_stopwords: {str(e)}")
            return tokens

    def lemmatize_text(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize the tokens to their root form.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: List of lemmatized tokens
        """
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception as e:
            self.logger.error(f"Error in lemmatize_text: {str(e)}")
            return tokens

    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline for input text.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Fully preprocessed text
        """
        try:
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Tokenize
            tokens = self.tokenize_text(cleaned_text)
            
            # Remove stopwords
            tokens_without_stopwords = self.remove_stopwords(tokens)
            
            # Lemmatize
            lemmatized_tokens = self.lemmatize_text(tokens_without_stopwords)
            
            # Join tokens back into text
            preprocessed_text = ' '.join(lemmatized_tokens)
            
            return preprocessed_text
        except Exception as e:
            self.logger.error(f"Error in preprocess_text: {str(e)}")
            return text

    def extract_features(self, text: str) -> List[str]:
        """
        Extract important features from the text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of important features
        """
        try:
            # Preprocess the text
            preprocessed_text = self.preprocess_text(text)
            
            # Extract features (for now, just return the preprocessed tokens)
            features = preprocessed_text.split()
            
            return features
        except Exception as e:
            self.logger.error(f"Error in extract_features: {str(e)}")
            return []
