"""
Newsletter Sentiment Analyzer

This module provides sentiment analysis capabilities for financial newsletters,
extracting sentiment scores and key phrases related to specific tickers.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class NewsletterSentimentAnalyzer:
    """
    Analyzes sentiment in financial newsletters with ticker-specific context.
    """
    
    def __init__(self, sentiment_terms_path: str = "config/sentiment_terms.json"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            sentiment_terms_path: Path to sentiment terms configuration
        """
        self.sentiment_terms = self._load_sentiment_terms(sentiment_terms_path)
        self.ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        
    def _load_sentiment_terms(self, path: str) -> Dict[str, Any]:
        """Load sentiment terms from configuration file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Sentiment terms file not found at {path}, using defaults")
            return self._get_default_sentiment_terms()
    
    def _get_default_sentiment_terms(self) -> Dict[str, Any]:
        """Get default sentiment terms if config file is not available."""
        return {
            "bullish_terms": [
                "bullish", "buy", "strong buy", "outperform", "positive", "growth",
                "rally", "surge", "breakout", "momentum", "uptrend", "gains",
                "profit", "revenue growth", "beat estimates", "upgrade", "target raised"
            ],
            "bearish_terms": [
                "bearish", "sell", "strong sell", "underperform", "negative", "decline",
                "crash", "drop", "breakdown", "weakness", "downtrend", "losses",
                "miss estimates", "downgrade", "target lowered", "risk", "concern"
            ],
            "amplifiers": [
                "very", "extremely", "highly", "significantly", "substantially",
                "dramatically", "strongly", "aggressively", "massive", "huge"
            ],
            "diminishers": [
                "slightly", "somewhat", "moderately", "mildly", "cautiously",
                "potentially", "possibly", "maybe", "might", "could"
            ]
        }
    
    def analyze_sentiment(self, text: str, ticker: str) -> Dict[str, Any]:
        """
        Analyze sentiment for a specific ticker in the given text.
        
        Args:
            text: Newsletter text to analyze
            ticker: Stock ticker to analyze sentiment for
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        # Clean and prepare text
        text_lower = text.lower()
        ticker_lower = ticker.lower()
        
        # Find ticker context
        context_found, context_text, context_length = self._extract_ticker_context(
            text, ticker, window_size=200
        )
        
        if not context_found:
            # Fallback to general sentiment if ticker not found
            context_text = text
            context_length = len(text)
        
        # Calculate sentiment scores
        sentiment_score = self._calculate_sentiment_score(context_text)
        confidence = self._calculate_confidence(context_text, ticker)
        key_phrases = self._extract_key_phrases(context_text, ticker)
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'key_phrases': key_phrases,
            'context_found': context_found,
            'context_length': context_length
        }
    
    def batch_analyze(self, text: str, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze sentiment for multiple tickers in the same text.
        
        Args:
            text: Newsletter text to analyze
            tickers: List of stock tickers to analyze
            
        Returns:
            Dictionary mapping tickers to their sentiment analysis results
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.analyze_sentiment(text, ticker)
        return results
    
    def _extract_ticker_context(self, text: str, ticker: str, 
                               window_size: int = 200) -> tuple:
        """
        Extract context around ticker mentions.
        
        Args:
            text: Full text to search
            ticker: Ticker symbol to find
            window_size: Number of characters around ticker mention
            
        Returns:
            Tuple of (context_found, context_text, context_length)
        """
        ticker_pattern = re.compile(rf'\b{re.escape(ticker)}\b', re.IGNORECASE)
        matches = list(ticker_pattern.finditer(text))
        
        if not matches:
            return False, "", 0
        
        # Combine contexts from all mentions
        contexts = []
        for match in matches:
            start = max(0, match.start() - window_size)
            end = min(len(text), match.end() + window_size)
            context = text[start:end].strip()
            contexts.append(context)
        
        # Join contexts and remove duplicates
        combined_context = " ... ".join(contexts)
        
        return True, combined_context, len(combined_context)
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """
        Calculate sentiment score for the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1 (bearish) and 1 (bullish)
        """
        text_lower = text.lower()
        
        # Count sentiment terms
        bullish_count = sum(1 for term in self.sentiment_terms["bullish_terms"] 
                           if term in text_lower)
        bearish_count = sum(1 for term in self.sentiment_terms["bearish_terms"] 
                           if term in text_lower)
        
        # Count amplifiers and diminishers
        amplifier_count = sum(1 for term in self.sentiment_terms["amplifiers"] 
                             if term in text_lower)
        diminisher_count = sum(1 for term in self.sentiment_terms["diminishers"] 
                              if term in text_lower)
        
        # Calculate base sentiment
        total_sentiment_terms = bullish_count + bearish_count
        if total_sentiment_terms == 0:
            base_sentiment = 0.0
        else:
            base_sentiment = (bullish_count - bearish_count) / total_sentiment_terms
        
        # Apply amplification/diminishing effects
        amplification_factor = 1.0 + (amplifier_count * 0.2) - (diminisher_count * 0.1)
        amplification_factor = max(0.1, min(2.0, amplification_factor))  # Clamp between 0.1 and 2.0
        
        sentiment_score = base_sentiment * amplification_factor
        
        # Clamp final score between -1 and 1
        return max(-1.0, min(1.0, sentiment_score))
    
    def _calculate_confidence(self, text: str, ticker: str) -> float:
        """
        Calculate confidence score for the sentiment analysis.
        
        Args:
            text: Text that was analyzed
            ticker: Ticker symbol
            
        Returns:
            Confidence score between 0 and 1
        """
        text_lower = text.lower()
        ticker_lower = ticker.lower()
        
        # Base confidence factors
        confidence_factors = []
        
        # Factor 1: Presence of ticker
        ticker_mentions = len(re.findall(rf'\b{re.escape(ticker)}\b', text, re.IGNORECASE))
        ticker_factor = min(1.0, ticker_mentions / 3.0)  # Max confidence at 3+ mentions
        confidence_factors.append(ticker_factor)
        
        # Factor 2: Number of sentiment terms
        total_sentiment_terms = (
            sum(1 for term in self.sentiment_terms["bullish_terms"] if term in text_lower) +
            sum(1 for term in self.sentiment_terms["bearish_terms"] if term in text_lower)
        )
        sentiment_factor = min(1.0, total_sentiment_terms / 5.0)  # Max confidence at 5+ terms
        confidence_factors.append(sentiment_factor)
        
        # Factor 3: Text length (more text generally means more reliable analysis)
        length_factor = min(1.0, len(text) / 500.0)  # Max confidence at 500+ characters
        confidence_factors.append(length_factor)
        
        # Factor 4: Presence of financial keywords
        financial_keywords = [
            "earnings", "revenue", "profit", "loss", "guidance", "forecast",
            "analyst", "rating", "target", "price", "valuation", "market"
        ]
        financial_mentions = sum(1 for keyword in financial_keywords if keyword in text_lower)
        financial_factor = min(1.0, financial_mentions / 3.0)
        confidence_factors.append(financial_factor)
        
        # Calculate weighted average confidence
        weights = [0.3, 0.3, 0.2, 0.2]  # Ticker and sentiment terms are most important
        confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
        
        return max(0.1, min(1.0, confidence))  # Ensure confidence is between 0.1 and 1.0
    
    def _extract_key_phrases(self, text: str, ticker: str) -> List[str]:
        """
        Extract key phrases related to the ticker and sentiment.
        
        Args:
            text: Text to extract phrases from
            ticker: Ticker symbol
            
        Returns:
            List of key phrases
        """
        key_phrases = []
        text_lower = text.lower()
        
        # Find sentences containing the ticker
        sentences = re.split(r'[.!?]+', text)
        ticker_sentences = [
            sentence.strip() for sentence in sentences 
            if re.search(rf'\b{re.escape(ticker)}\b', sentence, re.IGNORECASE)
        ]
        
        # Extract phrases with sentiment terms
        all_sentiment_terms = (
            self.sentiment_terms["bullish_terms"] + 
            self.sentiment_terms["bearish_terms"]
        )
        
        for sentence in ticker_sentences:
            sentence_lower = sentence.lower()
            for term in all_sentiment_terms:
                if term in sentence_lower:
                    # Extract phrase around the sentiment term
                    phrase = self._extract_phrase_around_term(sentence, term)
                    if phrase and len(phrase) > 10:  # Only include meaningful phrases
                        key_phrases.append(phrase)
        
        # Remove duplicates and limit to top 5 phrases
        unique_phrases = list(dict.fromkeys(key_phrases))  # Preserve order while removing duplicates
        return unique_phrases[:5]
    
    def _extract_phrase_around_term(self, sentence: str, term: str, 
                                   window_words: int = 5) -> str:
        """
        Extract a phrase around a specific term.
        
        Args:
            sentence: Sentence containing the term
            term: Term to extract phrase around
            window_words: Number of words to include on each side
            
        Returns:
            Extracted phrase
        """
        words = sentence.split()
        term_lower = term.lower()
        
        for i, word in enumerate(words):
            if term_lower in word.lower():
                start = max(0, i - window_words)
                end = min(len(words), i + window_words + 1)
                phrase = " ".join(words[start:end])
                return phrase.strip()
        
        return sentence.strip()
