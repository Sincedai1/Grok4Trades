"""
Sentiment Analysis for QuantumSol Stack

This module provides sentiment analysis capabilities using OpenAI's models
to analyze market news, social media, and other text data for trading signals.
"""
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import openai
from openai.types.chat import ChatCompletionMessageParam
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
import aiohttp
import json
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class SentimentLabel(Enum):
    """Sentiment classification labels"""
    STRONGLY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    STRONGLY_BULLISH = 2

@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""
    score: float  # -1.0 (very bearish) to 1.0 (very bullish)
    label: SentimentLabel
    confidence: float  # 0.0 to 1.0
    keywords: List[Tuple[str, float]]  # List of (keyword, relevance) tuples
    entities: List[Dict[str, Any]]  # List of extracted entities
    summary: Optional[str] = None
    metadata: Dict[str, Any] = None

class SentimentAnalyzer:
    """
    Performs sentiment analysis on financial text using OpenAI's models.
    
    Features:
    - Sentiment scoring (-1.0 to 1.0)
    - Sentiment classification
    - Keyword extraction
    - Entity recognition
    - Multi-document analysis
    - Caching of results
    """
    
    def __init__(self, 
                model: str = "gpt-4o",
                cache_dir: str = "./sentiment_cache"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model: OpenAI model to use for analysis
            cache_dir: Directory to cache analysis results
        """
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Sentiment analysis prompt templates
        self.sentiment_prompt = """
        Analyze the following financial text and determine the sentiment 
        towards the specified cryptocurrency or market.
        
        CRYPTOCURRENCY: {symbol}
        TEXT: """"{text}"""
        
        Provide a sentiment score from -1.0 (very bearish) to 1.0 (very bullish),
        a sentiment label, confidence score, and extract key information.
        """
        
        self.keyword_prompt = """
        Extract the most important keywords and entities from the following 
        financial text that are relevant to cryptocurrency markets.
        
        Return a JSON object with:
        - keywords: List of (keyword, relevance) where relevance is 0-1
        - entities: List of entities with type (PERSON, ORG, CRYPTO, etc.)
        
        TEXT: """"{text}"""
        """
    
    async def analyze(self, 
                     text: str, 
                     symbol: str = "SOL",
                     use_cache: bool = True) -> SentimentResult:
        """
        Analyze sentiment of the given text.
        
        Args:
            text: Text to analyze
            symbol: Cryptocurrency symbol (e.g., 'SOL', 'BTC')
            use_cache: Whether to use cached results if available
            
        Returns:
            SentimentResult with analysis
        """
        # Check cache first
        cache_key = self._generate_cache_key(text, symbol)
        if use_cache and (cached := self._get_from_cache(cache_key)):
            return cached
        
        # Prepare messages for the model
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": """You are a financial sentiment analysis expert. 
                Analyze the sentiment of the given text towards the specified cryptocurrency.
                Be concise and focus on actionable insights for traders."""
            },
            {
                "role": "user",
                "content": self.sentiment_prompt.format(symbol=symbol, text=text[:4000])  # Truncate to fit context
            }
        ]
        
        try:
            # Get sentiment analysis from the model
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
                response_format={ "type": "json_object" }
            )
            
            # Parse the response
            result = self._parse_sentiment_response(response.choices[0].message.content)
            
            # Extract keywords and entities
            keyword_info = await self._extract_keywords_and_entities(text)
            
            # Combine results
            final_result = SentimentResult(
                score=result.get('score', 0.0),
                label=self._score_to_label(result.get('score', 0.0)),
                confidence=result.get('confidence', 0.8),
                keywords=keyword_info.get('keywords', []),
                entities=keyword_info.get('entities', []),
                summary=result.get('summary'),
                metadata={
                    'model': self.model,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol
                }
            )
            
            # Cache the result
            self._save_to_cache(cache_key, final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Return neutral sentiment on error
            return SentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.5,
                keywords=[],
                entities=[],
                metadata={
                    'error': str(e),
                    'model': self.model,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
    
    async def analyze_batch(self, 
                          texts: List[str], 
                          symbol: str = "SOL",
                          use_cache: bool = True) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            symbol: Cryptocurrency symbol
            use_cache: Whether to use cached results
            
        Returns:
            List of SentimentResult objects
        """
        return [await self.analyze(text, symbol, use_cache) for text in texts]
    
    async def _extract_keywords_and_entities(self, text: str) -> Dict[str, Any]:
        """Extract keywords and entities from text"""
        try:
            messages: List[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": "Extract important keywords and entities from financial text."
                },
                {
                    "role": "user",
                    "content": self.keyword_prompt.format(text=text[:4000])
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
                response_format={ "type": "json_object" }
            )
            
            try:
                data = json.loads(response.choices[0].message.content)
                return {
                    'keywords': data.get('keywords', []),
                    'entities': data.get('entities', [])
                }
            except json.JSONDecodeError:
                logger.warning("Failed to parse keyword extraction response")
                return {'keywords': [], 'entities': []}
                
        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}")
            return {'keywords': [], 'entities': []}
    
    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert a sentiment score to a label"""
        if score <= -0.6:
            return SentimentLabel.STRONGLY_BEARISH
        elif score <= -0.2:
            return SentimentLabel.BEARISH
        elif score <= 0.2:
            return SentimentLabel.NEUTRAL
        elif score <= 0.6:
            return SentimentLabel.BULLISH
        else:
            return SentimentLabel.STRONGLY_BULLISH
    
    def _generate_cache_key(self, text: str, symbol: str) -> str:
        """Generate a cache key for the given text and symbol"""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{symbol.lower()}_{text_hash[:16]}.json"
    
    def _get_from_cache(self, cache_key: str) -> Optional[SentimentResult]:
        """Get a result from cache if it exists and is recent"""
        cache_file = self.cache_dir / cache_key
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                
            # Check if cache is still valid (24 hours)
            cache_time = datetime.fromisoformat(data['metadata']['timestamp'])
            if (datetime.now(timezone.utc) - cache_time).days >= 1:
                return None
                
            # Convert back to SentimentResult
            return SentimentResult(
                score=data['score'],
                label=SentimentLabel(data['label']),
                confidence=data['confidence'],
                keywords=[(k, v) for k, v in data.get('keywords', [])],
                entities=data.get('entities', []),
                summary=data.get('summary'),
                metadata=data.get('metadata', {})
            )
            
        except Exception as e:
            logger.warning(f"Error reading from cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, result: SentimentResult):
        """Save a result to cache"""
        try:
            cache_file = self.cache_dir / cache_key
            
            # Convert to serializable format
            data = {
                'score': result.score,
                'label': result.label.value,
                'confidence': result.confidence,
                'keywords': [(k, float(v)) for k, v in result.keywords],
                'entities': result.entities,
                'summary': result.summary,
                'metadata': result.metadata
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

# Global instance
sentiment_analyzer = SentimentAnalyzer()

# Example usage:
# result = await sentiment_analyzer.analyze(
#     "Solana's price surged 15% today after the announcement of a new partnership with Visa.",
#     symbol="SOL"
# )
# print(f"Sentiment: {result.label.name} ({result.score:.2f})")
# print(f"Confidence: {result.confidence:.2f}")
# print(f"Keywords: {result.keywords}")
