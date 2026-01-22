"""
Text preprocessing utilities for CryptoSentinel.

This module provides text preprocessing functionality extracted from the original
NLP pipeline, including tokenization, lemmatization, and stopword removal.
"""

import re
from typing import List, Optional

import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """
    Text preprocessing pipeline for Reddit posts.
    
    Handles tokenization, lemmatization, lowercasing, punctuation removal,
    and stopword filtering for crypto/stock discussion text.
    """
    
    def __init__(self, custom_stopwords: Optional[List[str]] = None):
        """
        Initialize the text preprocessor.
        
        Args:
            custom_stopwords: Additional stopwords to filter out.
        """
        self._ensure_nltk_data()
        
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        self.crypto_terms = {
            'btc', 'eth', 'bitcoin', 'ethereum', 'crypto', 'blockchain',
            'defi', 'nft', 'altcoin', 'hodl', 'moon', 'pump', 'dump',
            'whale', 'fud', 'fomo', 'doge', 'shib', 'sol', 'ada'
        }
        
        self.stock_terms = {
            'stock', 'shares', 'options', 'calls', 'puts', 'yolo',
            'tendies', 'ape', 'diamond', 'hands', 'gme', 'amc',
            'spy', 'qqq', 'nasdaq', 'nyse', 'sec', 'earnings'
        }
    
    def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is downloaded."""
        required = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for item in required:
            try:
                nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else f'corpora/{item}')
            except LookupError:
                nltk.download(item, quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text by removing URLs, special characters, and extra whitespace.
        
        Args:
            text: Raw input text.
            
        Returns:
            Cleaned text string.
        """
        if not text or text == 'notexthere':
            return ''
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[^\w\s$]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip().lower()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string.
            
        Returns:
            List of tokens.
        """
        return self.tokenizer.tokenize(text.lower())
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list, preserving domain-specific terms.
        
        Args:
            tokens: List of word tokens.
            
        Returns:
            Filtered token list.
        """
        preserved = self.crypto_terms | self.stock_terms
        return [
            token for token in tokens
            if token not in self.stop_words or token in preserved
        ]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of word tokens.
            
        Returns:
            List of lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline: clean, tokenize, remove stopwords, lemmatize.
        
        Args:
            text: Raw input text.
            
        Returns:
            Preprocessed text string.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return ''
        
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        
        return ' '.join(tokens)
    
    def preprocess_title(self, title: str) -> str:
        """
        Preprocess a post title (lighter processing, preserve more context).
        
        Args:
            title: Post title string.
            
        Returns:
            Preprocessed title string.
        """
        if not title:
            return ''
        
        cleaned = self.clean_text(title)
        tokens = self.tokenize(cleaned)
        tokens = self.lemmatize(tokens)
        
        return ' '.join(tokens)
    
    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock/crypto ticker symbols from text.
        
        Args:
            text: Input text string.
            
        Returns:
            List of detected ticker symbols.
        """
        ticker_pattern = r'\$([A-Za-z]{1,5})\b'
        tickers = re.findall(ticker_pattern, text)
        return [t.upper() for t in tickers]
