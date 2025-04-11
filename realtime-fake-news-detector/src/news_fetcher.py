import requests
from newsapi import NewsApiClient
from typing import List, Dict, Optional
import logging

class NewsAPI:
    def __init__(self, api_key: str):
        """Initialize NewsAPI with the provided API key."""
        try:
            self.api = NewsApiClient(api_key=api_key)
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            self.logger.error(f"Failed to initialize NewsAPI: {str(e)}")
            raise

    def fetch_latest_news(self, language: str = 'en', page_size: int = 10) -> List[Dict]:
        """
        Fetch the latest news articles.
        
        Args:
            language (str): Language of news articles (default: 'en')
            page_size (int): Number of articles to fetch (default: 10)
            
        Returns:
            List[Dict]: List of news articles
        """
        try:
            response = self.api.get_top_headlines(
                language=language,
                page_size=page_size
            )
            
            if response['status'] == 'ok':
                return response['articles']
            else:
                self.logger.error(f"Failed to fetch news: {response['message']}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching latest news: {str(e)}")
            return []

    def get_news_by_source(self, source: str, page_size: int = 10) -> List[Dict]:
        """
        Fetch news from a specific source.
        
        Args:
            source (str): News source ID
            page_size (int): Number of articles to fetch
            
        Returns:
            List[Dict]: List of news articles
        """
        try:
            response = self.api.get_everything(
                sources=source,
                page_size=page_size,
                language='en'
            )
            
            if response['status'] == 'ok':
                return response['articles']
            else:
                self.logger.error(f"Failed to fetch news by source: {response['message']}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching news by source: {str(e)}")
            return []

    def get_news_by_topic(self, topic: str, page_size: int = 10) -> List[Dict]:
        """
        Fetch news related to a specific topic.
        
        Args:
            topic (str): Topic to search for
            page_size (int): Number of articles to fetch
            
        Returns:
            List[Dict]: List of news articles
        """
        try:
            response = self.api.get_everything(
                q=topic,
                page_size=page_size,
                language='en',
                sort_by='relevancy'
            )
            
            if response['status'] == 'ok':
                return response['articles']
            else:
                self.logger.error(f"Failed to fetch news by topic: {response['message']}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching news by topic: {str(e)}")
            return []

    def verify_url(self, url: str) -> bool:
        """
        Verify if a news URL is accessible.
        
        Args:
            url (str): URL to verify
            
        Returns:
            bool: True if URL is accessible, False otherwise
        """
        try:
            response = requests.head(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Error verifying URL {url}: {str(e)}")
            return False
