""
Sentiment analysis for crypto assets using multiple data sources.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import re
import json

import aiohttp
import numpy as np
from pydantic import BaseModel, Field, validator
from textblob import TextBlob
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

class SentimentSource(str, Enum):
    """Sources for sentiment data."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    COINGECKO = "coingecko"
    COINMARKETCAP = "coinmarketcap"
    ONCHAIN = "onchain"
    
class SentimentScore(BaseModel):
    """Sentiment score for a given asset."""
    score: float = Field(..., ge=-1.0, le=1.0)  # -1.0 (very bearish) to 1.0 (very bullish)
    magnitude: float = Field(..., ge=0.0)  # Strength of sentiment
    source: SentimentSource
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(..., ge=0.0, le=1.0)  # Confidence in the score
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('score')
    def validate_score(cls, v):
        return round(v, 4)
    
    @validator('magnitude')
    def validate_magnitude(cls, v):
        return round(v, 4)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return round(v, 4)

class SentimentAnalyzer:
    """Analyzes sentiment from various sources for crypto assets."""
    
    def __init__(self, model_name: str = "finiteautomata/bertweet-base-sentiment-analysis"):
        """Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the HuggingFace model to use for sentiment analysis.
        """
        self.model_name = model_name
        self.sia = SentimentIntensityAnalyzer()
        self.classifier = None
        self.session = None
        
        # Initialize the classifier in the background
        self._init_task = asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the sentiment analysis model."""
        try:
            # Initialize the classifier
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                framework="pt"
            )
            
            # Create a session for HTTP requests
            self.session = aiohttp.ClientSession()
            
            logger.info(f"Initialized sentiment analyzer with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            raise
    
    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def analyze_text(self, text: str, source: SentimentSource) -> SentimentScore:
        """Analyze sentiment of a single text.
        
        Args:
            text: The text to analyze.
            source: The source of the text.
            
        Returns:
            SentimentScore: The sentiment score for the text.
        """
        # Wait for initialization to complete
        if self._init_task and not self._init_task.done():
            await self._init_task
        
        if not text or not text.strip():
            return SentimentScore(
                score=0.0,
                magnitude=0.0,
                source=source,
                confidence=0.0,
                metadata={"error": "Empty text"}
            )
        
        try:
            # Clean the text
            text = self._clean_text(text)
            
            # Get sentiment from TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Get sentiment from VADER
            vader_scores = self.sia.polarity_scores(text)
            vader_compound = vader_scores['compound']
            
            # Get sentiment from the transformer model
            if self.classifier:
                result = self.classifier(text, truncation=True, max_length=512)
                model_score = self._transform_model_output(result)
            else:
                model_score = 0.0
            
            # Combine scores (weighted average)
            weights = {
                'textblob': 0.3,
                'vader': 0.4,
                'transformer': 0.3
            }
            
            # Calculate weighted score
            score = (
                weights['textblob'] * polarity +
                weights['vader'] * vader_compound +
                weights['transformer'] * model_score
            )
            
            # Calculate magnitude (strength of sentiment)
            magnitude = abs(score)
            
            # Calculate confidence (higher for more extreme scores and more subjective text)
            confidence = min(1.0, (abs(score) * 0.7) + (subjectivity * 0.3))
            
            return SentimentScore(
                score=score,
                magnitude=magnitude,
                source=source,
                confidence=confidence,
                metadata={
                    "textblob_score": polarity,
                    "vader_score": vader_compound,
                    "transformer_score": model_score,
                    "subjectivity": subjectivity,
                    "processed_text": text[:500]  # Store first 500 chars for debugging
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}", exc_info=True)
            return SentimentScore(
                score=0.0,
                magnitude=0.0,
                source=source,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def analyze_multiple_texts(
        self,
        texts: List[str],
        source: SentimentSource
    ) -> Tuple[SentimentScore, List[SentimentScore]]:
        """Analyze sentiment of multiple texts and return an aggregate score.
        
        Args:
            texts: List of texts to analyze.
            source: The source of the texts.
            
        Returns:
            Tuple of (aggregate_score, individual_scores)
        """
        if not texts:
            return (
                SentimentScore(
                    score=0.0,
                    magnitude=0.0,
                    source=source,
                    confidence=0.0,
                    metadata={"error": "No texts provided"}
                ),
                []
            )
        
        # Analyze each text
        tasks = [self.analyze_text(text, source) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out any errors
        valid_scores = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing text {i}: {result}")
            elif isinstance(result, SentimentScore):
                valid_scores.append(result)
        
        if not valid_scores:
            return (
                SentimentScore(
                    score=0.0,
                    magnitude=0.0,
                    source=source,
                    confidence=0.0,
                    metadata={"error": "No valid scores"}
                ),
                []
            )
        
        # Calculate weighted average score based on confidence
        total_weight = sum(score.confidence for score in valid_scores)
        if total_weight > 0:
            weighted_score = sum(score.score * score.confidence for score in valid_scores) / total_weight
            avg_magnitude = sum(score.magnitude for score in valid_scores) / len(valid_scores)
            avg_confidence = sum(score.confidence for score in valid_scores) / len(valid_scores)
        else:
            weighted_score = 0.0
            avg_magnitude = 0.0
            avg_confidence = 0.0
        
        # Create aggregate score
        aggregate = SentimentScore(
            score=weighted_score,
            magnitude=avg_magnitude,
            source=source,
            confidence=avg_confidence,
            metadata={
                "num_texts": len(texts),
                "num_valid_scores": len(valid_scores)
            }
        )
        
        return aggregate, valid_scores
    
    async def get_twitter_sentiment(
        self,
        query: str,
        limit: int = 100,
        lookback_hours: int = 24
    ) -> Tuple[SentimentScore, List[Dict]]:
        """Get sentiment from Twitter for a given query.
        
        Args:
            query: Search query (e.g., "$SOL", "#Bitcoin").
            limit: Maximum number of tweets to analyze.
            lookback_hours: How far back to search in hours.
            
        Returns:
            Tuple of (aggregate_score, tweets_with_sentiment)
        """
        # This is a placeholder implementation
        # In a real implementation, you would use the Twitter API
        logger.warning("Twitter sentiment analysis requires API integration")
        
        # Return a neutral score
        return (
            SentimentScore(
                score=0.0,
                magnitude=0.0,
                source=SentimentSource.TWITTER,
                confidence=0.0,
                metadata={"error": "Twitter API not implemented"}
            ),
            []
        )
    
    async def get_reddit_sentiment(
        self,
        subreddit: str,
        query: Optional[str] = None,
        limit: int = 50,
        lookback_hours: int = 24
    ) -> Tuple[SentimentScore, List[Dict]]:
        """Get sentiment from Reddit for a given subreddit and optional query.
        
        Args:
            subreddit: Subreddit to search (e.g., "solana", "CryptoCurrency").
            query: Optional search query.
            limit: Maximum number of posts/comments to analyze.
            lookback_hours: How far back to search in hours.
            
        Returns:
            Tuple of (aggregate_score, posts_with_sentiment)
        """
        # This is a placeholder implementation
        logger.warning("Reddit sentiment analysis requires API integration")
        
        # Return a neutral score
        return (
            SentimentScore(
                score=0.0,
                magnitude=0.0,
                source=SentimentSource.REDDIT,
                confidence=0.0,
                metadata={"error": "Reddit API not implemented"}
            ),
            []
        )
    
    async def get_news_sentiment(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        limit: int = 20,
        lookback_hours: int = 24
    ) -> Tuple[SentimentScore, List[Dict]]:
        """Get sentiment from news articles.
        
        Args:
            query: Search query (e.g., "Solana", "Ethereum").
            sources: Optional list of news sources to filter by.
            limit: Maximum number of articles to analyze.
            lookback_hours: How far back to search in hours.
            
        Returns:
            Tuple of (aggregate_score, articles_with_sentiment)
        """
        # This is a placeholder implementation
        logger.warning("News sentiment analysis requires API integration")
        
        # Return a neutral score
        return (
            SentimentScore(
                score=0.0,
                magnitude=0.0,
                source=SentimentSource.NEWS,
                confidence=0.0,
                metadata={"error": "News API not implemented"}
            ),
            []
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis."""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove special characters and numbers (keep letters, basic punctuation)
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def _transform_model_output(self, output) -> float:
        """Transform model output to a sentiment score between -1 and 1."""
        if not output:
            return 0.0
        
        # Handle different output formats
        if isinstance(output, list) and output:
            item = output[0]
            if isinstance(item, dict):
                label = item.get('label', '').lower()
                score = item.get('score', 0.0)
                
                # Map label to score
                if 'negative' in label:
                    return -score
                elif 'positive' in label:
                    return score
                elif 'neutral' in label:
                    return 0.0
                else:
                    # For binary classification (pos/neg)
                    return score if 'pos' in label else -score
        
        return 0.0

# Global instance
sentiment_analyzer = SentimentAnalyzer()

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get the global sentiment analyzer instance."""
    return sentiment_analyzer
