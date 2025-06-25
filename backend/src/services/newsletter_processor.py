import re
import requests
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import html2text
import os
from src.services.cache_service import CacheService

logger = logging.getLogger(__name__)

class NewsletterProcessor:
    """
    Service for processing trading newsletters and extracting trading signals.
    Handles email parsing, content extraction, and ticker identification.
    """
    
    def __init__(self, ml_engine_url: str = None):
        self.ml_engine_url = ml_engine_url or os.getenv("ML_ENGINE_URL", "http://localhost:8000")
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.cache_service = CacheService()
        
        # Common ticker patterns and exchanges
        self.ticker_patterns = [
            r"\b[A-Z]{1,5}\b",  # Basic ticker pattern
            r"\$[A-Z]{1,5}\b",  # Ticker with $ prefix
            r"\b[A-Z]{1,5}:[A-Z]{2,3}\b",  # Exchange:Ticker format
        ]
        
        # Known exchanges and suffixes
        self.exchanges = ["NYSE", "NASDAQ", "AMEX", "OTC"]
        self.ticker_suffixes = [".TO", ".L", ".HK", ".SS", ".SZ"]
        
        # Common false positives to filter out
        self.false_positives = {
            "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE",
            "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW", "ITS", "NEW", "NOW", "OLD",
            "SEE", "TWO", "WHO", "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE",
            "CEO", "CFO", "CTO", "IPO", "SEC", "FDA", "FTC", "DOJ", "IRS", "GDP", "CPI", "PPI",
            "ETF", "ETN", "REITs", "REIT", "LLC", "INC", "CORP", "LTD", "USA", "USD", "EUR",
            "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "API", "URL", "HTTP", "HTTPS", "WWW",
            "PDF", "CSV", "XML", "JSON", "HTML", "CSS", "SQL", "AWS", "IBM", "AI", "ML", "IT"
        }
        
        # Known ticker symbols for validation
        self.known_tickers = {
            # Major stocks
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA", "BRK.A", "BRK.B",
            "JNJ", "V", "WMT", "PG", "JPM", "UNH", "MA", "HD", "DIS", "ADBE", "NFLX", "CRM",
            "BAC", "ABBV", "PFE", "KO", "PEP", "TMO", "COST", "AVGO", "DHR", "NEE", "XOM",
            "LLY", "ABT", "ACN", "TXN", "NKE", "CVX", "QCOM", "MCD", "LIN", "HON", "UPS",
            "LOW", "ORCL", "AMD", "INTC", "IBM", "INTU", "CAT", "GS", "AXP", "RTX", "SPGI",
            "NOW", "ISRG", "BKNG", "TGT", "DE", "SYK", "ZTS", "BLK", "MDLZ", "GILD", "ADP",
            "VRTX", "LRCX", "ADI", "REGN", "PYPL", "SCHW", "TMUS", "MU", "AMAT", "KLAC",
            
            # ETFs
            "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND", "TLT", "GLD",
            "SLV", "USO", "UNG", "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
            "XLB", "XLRE", "XBI", "SMH", "SOXX", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF",
            
            # Crypto-related
            "COIN", "MSTR", "RIOT", "MARA", "HUT", "BITF", "CAN", "HIVE", "BTBT", "SOS"
        }
    
    def extract_content_from_email(self, email_content: str, content_type: str = "html") -> Dict:
        """
        Extract and clean content from email.
        
        Args:
            email_content: Raw email content
            content_type: "html" or "text"
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            if content_type.lower() == "html":
                # Convert HTML to text
                text_content = self.html_converter.handle(email_content)
            else:
                text_content = email_content
            
            # Clean the text
            cleaned_content = self._clean_text(text_content)
            
            # Extract metadata
            metadata = self._extract_metadata(cleaned_content)
            
            return {
                "content": cleaned_content,
                "word_count": len(cleaned_content.split()),
                "char_count": len(cleaned_content),
                "metadata": metadata,
                "extraction_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error extracting email content: {str(e)}")
            return {
                "content": "",
                "word_count": 0,
                "char_count": 0,
                "metadata": {},
                "error": str(e)
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Remove email headers and footers
        text = re.sub(r"^.*?Subject:.*?\n", "", text, flags=re.MULTILINE | re.DOTALL)
        text = re.sub(r"Unsubscribe.*?$", "", text, flags=re.MULTILINE | re.DOTALL)
        text = re.sub(r"This email was sent.*?$", "", text, flags=re.MULTILINE | re.DOTALL)
        
        # Remove URLs
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)
        
        # Remove email addresses
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
        
        # Remove excessive punctuation
        text = re.sub(r"[.]{3,}", "...", text)
        text = re.sub(r"[-]{3,}", "---", text)
        
        # Clean up spacing
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = text.strip()
        
        return text
    
    def _extract_metadata(self, content: str) -> Dict:
        """Extract metadata from newsletter content."""
        metadata = {}
        
        # Extract publication info
        pub_patterns = [
            r"Published by ([^\\n]+)",
            r"From: ([^\\n]+)",
            r"Author: ([^\\n]+)"
        ]
        
        for pattern in pub_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["publisher"] = match.group(1).strip()
                break
        
        # Extract date patterns
        date_patterns = [
            r"(\d{1,2}/\d{1,2}/\d{4})",
            r"(\d{4}-\d{2}-\d{2})",
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["publication_date"] = match.group(1)
                break
        
        # Extract newsletter type indicators
        if any(word in content.lower() for word in ["morning", "daily", "am"]):
            metadata["newsletter_type"] = "morning"
        elif any(word in content.lower() for word in ["evening", "close", "pm"]):
            metadata["newsletter_type"] = "evening"
        elif any(word in content.lower() for word in ["weekly", "week"]):
            metadata["newsletter_type"] = "weekly"
        
        return metadata
    
    def extract_tickers(self, content: str) -> List[str]:
        """
        Extract stock tickers from newsletter content.
        
        Args:
            content: Newsletter text content
            
        Returns:
            List of unique ticker symbols found
        """
        try:
            tickers = set()
            
            # Apply each ticker pattern
            for pattern in self.ticker_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    # Clean the ticker
                    ticker = match.replace("$", "").strip()
                    
                    # Validate ticker
                    if self._is_valid_ticker(ticker):
                        tickers.add(ticker)
            
            # Sort by frequency in text (most mentioned first)
            ticker_counts = {}
            for ticker in tickers:
                ticker_counts[ticker] = len(re.findall(rf"\\b{re.escape(ticker)}\\b", content, re.IGNORECASE))
            
            # Return sorted by frequency
            sorted_tickers = sorted(ticker_counts.keys(), key=lambda x: ticker_counts[x], reverse=True)
            
            logger.info(f"Extracted {len(sorted_tickers)} tickers: {sorted_tickers[:10]}")
            return sorted_tickers
            
        except Exception as e:
            logger.error(f"Error extracting tickers: {str(e)}")
            return []
    
    def _is_valid_ticker(self, ticker: str) -> bool:
        """
        Validate if a string is likely a valid ticker symbol.
        This method can be extended to integrate with external APIs for real-time validation.
        """
        # Basic length check
        if len(ticker) < 1 or len(ticker) > 5:
            return False
        
        # Must be all uppercase letters
        if not ticker.isupper() or not ticker.isalpha():
            return False
        
        # Filter out common false positives
        if ticker in self.false_positives:
            return False
        
        # Prefer known tickers
        if ticker in self.known_tickers:
            return True
        
        # Additional validation for unknown tickers
        # Must be 2-5 characters for unknown tickers
        if len(ticker) < 2:
            return False
        
        # Avoid common English words
        common_words = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE"}
        if ticker in common_words:
            return False
        
        return True
    
    def analyze_sentiment_for_tickers(self, content: str, tickers: List[str]) -> Dict[str, Dict]:
        """
        Analyze sentiment for each ticker using the ML engine.
        
        Args:
            content: Newsletter content
            tickers: List of tickers to analyze
            
        Returns:
            Dictionary mapping tickers to sentiment analysis results
        """
        try:
            if not tickers:
                return {}
            
            # Generate a cache key based on content hash and tickers
            cache_key = f"sentiment_{hash(content)}_{hash(frozenset(tickers))}"
            cached_result = self.cache_service.get(cache_key)
            if cached_result:
                logger.info(f"Returning cached sentiment for {tickers}")
                return cached_result

            # Prepare batch request for ML engine
            payload = {
                "text": content,
                "tickers": tickers
            }
            
            response = requests.post(
                f"{self.ml_engine_url}/sentiment/batch",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                sentiment_results = response.json()
                self.cache_service.set(cache_key, sentiment_results, ttl_seconds=3600) # Cache for 1 hour
                return sentiment_results
            else:
                logger.error(f"ML engine sentiment analysis failed: {response.status_code} - {response.text}")
                return {}
                
        except requests.RequestException as e:
            logger.error(f"Error calling ML engine for sentiment analysis: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in sentiment analysis: {str(e)}")
            return {}
    
    def process_newsletter(self, email_content: str, subject: str = "", source: str = "", content_type: str = "html") -> Dict:
        """
        Complete newsletter processing pipeline.
        
        Args:
            email_content: Raw email content
            subject: Email subject line
            source: Newsletter source/sender
            content_type: "html" or "text"
            
        Returns:
            Complete analysis results
        """
        try:
            logger.info(f"Processing newsletter from {source}")
            
            # Extract and clean content
            extraction_result = self.extract_content_from_email(email_content, content_type)
            
            if not extraction_result["content"]:
                return {
                    "success": False,
                    "error": "Failed to extract content from email"
                }
            
            content = extraction_result["content"]
            
            # Extract tickers
            tickers = self.extract_tickers(content)
            
            if not tickers:
                logger.warning("No tickers found in newsletter content")
                return {
                    "success": True,
                    "content": content,
                    "subject": subject,
                    "source": source,
                    "tickers": [],
                    "sentiment_analysis": {},
                    "metadata": extraction_result["metadata"],
                    "warning": "No tickers found"
                }
            
            # Analyze sentiment for each ticker
            sentiment_results = self.analyze_sentiment_for_tickers(content, tickers)
            
            # Calculate overall newsletter sentiment
            overall_sentiment = self._calculate_overall_sentiment(sentiment_results)
            
            # Determine priority score
            priority_score = self._calculate_priority_score(tickers, sentiment_results, source)
            
            result = {
                "success": True,
                "content": content,
                "subject": subject,
                "source": source,
                "tickers": tickers,
                "sentiment_analysis": sentiment_results,
                "overall_sentiment": overall_sentiment,
                "priority_score": priority_score,
                "metadata": extraction_result["metadata"],
                "processing_timestamp": datetime.utcnow().isoformat(),
                "word_count": extraction_result["word_count"],
                "ticker_count": len(tickers)
            }
            
            logger.info(f"Successfully processed newsletter: {len(tickers)} tickers, priority {priority_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing newsletter: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_timestamp": datetime.utcnow().isoformat()
            }
    
    def _calculate_overall_sentiment(self, sentiment_results: Dict[str, Dict]) -> Dict:
        """
        Calculate overall sentiment across all tickers.
        """
        if not sentiment_results:
            return {"score": 0.0, "confidence": 0.0}
        
        total_score = 0
        total_confidence = 0
        valid_results = 0
        
        for ticker, result in sentiment_results.items():
            if result.get("context_found", False):
                total_score += result.get("sentiment_score", 0)
                total_confidence += result.get("confidence", 0)
                valid_results += 1
        
        if valid_results == 0:
            return {"score": 0.0, "confidence": 0.0}
        
        return {
            "score": total_score / valid_results,
            "confidence": total_confidence / valid_results,
            "ticker_count": valid_results
        }
    
    def _calculate_priority_score(self, tickers: List[str], sentiment_results: Dict[str, Dict], source: str) -> int:
        """
        Calculate priority score for the newsletter (1-10).
        """
        base_score = 5
        
        # Adjust based on number of tickers
        if len(tickers) >= 5:
            base_score += 2
        elif len(tickers) >= 3:
            base_score += 1
        elif len(tickers) == 0:
            base_score -= 3
        
        # Adjust based on sentiment strength
        if sentiment_results:
            avg_confidence = sum(r.get("confidence", 0) for r in sentiment_results.values()) / len(sentiment_results)
            avg_sentiment_strength = sum(abs(r.get("sentiment_score", 0)) for r in sentiment_results.values()) / len(sentiment_results)
            
            if avg_confidence > 0.8 and avg_sentiment_strength > 0.5:
                base_score += 2
            elif avg_confidence > 0.6 and avg_sentiment_strength > 0.3:
                base_score += 1
        
        # Adjust based on source credibility (would be configurable)
        source_multipliers = {
            "pr

