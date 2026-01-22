"""
Classification models for CryptoSentinel.

Wraps the original Naive Bayes + Logistic Regression ensemble model
with production-ready interfaces for prediction and explanation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..utils.preprocessing import TextPreprocessor


@dataclass
class FeatureExplanation:
    """
    Explanation of feature contributions to a prediction.
    
    Attributes:
        feature: Feature name (word/n-gram).
        weight: Feature weight/importance.
        direction: "crypto" or "stocks" indicating which class it supports.
    """
    feature: str
    weight: float
    direction: str


@dataclass 
class ClassificationResult:
    """
    Result of classifying a post.
    
    Attributes:
        predicted_class: Predicted class ("crypto" or "stocks").
        confidence: Prediction confidence (0-1).
        probabilities: Class probabilities.
        explanations: Top contributing features.
    """
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    explanations: List[FeatureExplanation]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in self.probabilities.items()},
            "top_features": [
                {"feature": e.feature, "weight": round(e.weight, 4), "direction": e.direction}
                for e in self.explanations[:5]
            ]
        }


class EnsembleClassifier:
    """
    Ensemble classifier combining Naive Bayes and Logistic Regression.
    
    Replicates the original model architecture achieving 93%+ accuracy
    on crypto vs. stock post classification.
    """
    
    CLASS_NAMES = {0: "crypto", 1: "stocks"}
    CLASS_LABELS = {"crypto": 0, "stocks": 1}
    
    def __init__(
        self,
        nb_weight: float = 0.5,
        lr_weight: float = 0.5,
        vectorizer_type: str = "tfidf"
    ):
        """
        Initialize the ensemble classifier.
        
        Args:
            nb_weight: Weight for Naive Bayes predictions.
            lr_weight: Weight for Logistic Regression predictions.
            vectorizer_type: "tfidf" or "count" vectorization.
        """
        self.nb_weight = nb_weight
        self.lr_weight = lr_weight
        self.vectorizer_type = vectorizer_type
        
        self.preprocessor = TextPreprocessor()
        
        if vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_df=0.5,
                ngram_range=(1, 1),
                stop_words='english'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_df=0.5,
                ngram_range=(1, 1),
                stop_words='english'
            )
        
        self.nb_model = MultinomialNB(alpha=1.0)
        self.lr_model = LogisticRegression(C=10, max_iter=1000)
        
        self._fitted = False
        self._feature_names: List[str] = []
    
    def fit(
        self,
        texts: List[str],
        labels: List[int],
        preprocess: bool = True
    ) -> "EnsembleClassifier":
        """
        Fit the ensemble classifier.
        
        Args:
            texts: Training text samples.
            labels: Training labels (0=crypto, 1=stocks).
            preprocess: Whether to preprocess texts.
            
        Returns:
            Self for method chaining.
        """
        if preprocess:
            texts = [self.preprocessor.preprocess(t) for t in texts]
        
        X = self.vectorizer.fit_transform(texts)
        self._feature_names = self.vectorizer.get_feature_names_out().tolist()
        
        self.nb_model.fit(X, labels)
        self.lr_model.fit(X, labels)
        
        self._fitted = True
        return self
    
    def predict(self, text: str) -> ClassificationResult:
        """
        Predict class for a single text.
        
        Args:
            text: Input text to classify.
            
        Returns:
            ClassificationResult with prediction and explanations.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        processed = self.preprocessor.preprocess(text)
        X = self.vectorizer.transform([processed])
        
        nb_proba = self.nb_model.predict_proba(X)[0]
        lr_proba = self.lr_model.predict_proba(X)[0]
        
        ensemble_proba = (
            self.nb_weight * nb_proba + 
            self.lr_weight * lr_proba
        )
        
        predicted_idx = int(np.argmax(ensemble_proba))
        confidence = float(ensemble_proba[predicted_idx])
        
        explanations = self._explain_prediction(X, predicted_idx)
        
        return ClassificationResult(
            predicted_class=self.CLASS_NAMES[predicted_idx],
            confidence=confidence,
            probabilities={
                "crypto": float(ensemble_proba[0]),
                "stocks": float(ensemble_proba[1])
            },
            explanations=explanations
        )
    
    def predict_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Predict classes for multiple texts.
        
        Args:
            texts: List of input texts.
            
        Returns:
            List of ClassificationResult objects.
        """
        return [self.predict(text) for text in texts]
    
    def predict_with_confidence(
        self,
        text: str
    ) -> Tuple[str, float]:
        """
        Simple prediction returning class and confidence.
        
        Args:
            text: Input text to classify.
            
        Returns:
            Tuple of (predicted_class, confidence).
        """
        result = self.predict(text)
        return result.predicted_class, result.confidence
    
    def _explain_prediction(
        self,
        X,
        predicted_class: int,
        top_k: int = 10
    ) -> List[FeatureExplanation]:
        """
        Extract top contributing features for a prediction.
        
        Args:
            X: Vectorized input.
            predicted_class: Predicted class index.
            top_k: Number of top features to return.
            
        Returns:
            List of FeatureExplanation objects.
        """
        if hasattr(self.lr_model, 'coef_'):
            coef = self.lr_model.coef_[0]
        else:
            log_prob_diff = self.nb_model.feature_log_prob_[1] - self.nb_model.feature_log_prob_[0]
            coef = log_prob_diff
        
        feature_vector = X.toarray()[0]
        active_indices = np.nonzero(feature_vector)[0]
        
        contributions = []
        for idx in active_indices:
            weight = coef[idx] * feature_vector[idx]
            contributions.append((idx, weight))
        
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanations = []
        for idx, weight in contributions[:top_k]:
            feature_name = self._feature_names[idx]
            direction = "stocks" if weight > 0 else "crypto"
            explanations.append(FeatureExplanation(
                feature=feature_name,
                weight=abs(weight),
                direction=direction
            ))
        
        return explanations
    
    def explain_prediction(self, text: str, top_k: int = 10) -> List[FeatureExplanation]:
        """
        Get feature explanations for a text prediction.
        
        Args:
            text: Input text.
            top_k: Number of top features to return.
            
        Returns:
            List of FeatureExplanation objects.
        """
        result = self.predict(text)
        return result.explanations[:top_k]
    
    def get_feature_importance(self, top_k: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get overall feature importance for both classes.
        
        Args:
            top_k: Number of top features per class.
            
        Returns:
            Dictionary with crypto and stocks top features.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if hasattr(self.lr_model, 'coef_'):
            coef = self.lr_model.coef_[0]
        else:
            coef = self.nb_model.feature_log_prob_[1] - self.nb_model.feature_log_prob_[0]
        
        indices_sorted = np.argsort(coef)
        
        crypto_features = []
        for idx in indices_sorted[:top_k]:
            crypto_features.append({
                "feature": self._feature_names[idx],
                "weight": float(-coef[idx])
            })
        
        stocks_features = []
        for idx in indices_sorted[-top_k:][::-1]:
            stocks_features.append({
                "feature": self._feature_names[idx],
                "weight": float(coef[idx])
            })
        
        return {
            "crypto": crypto_features,
            "stocks": stocks_features
        }
    
    def save(self, path: Path) -> None:
        """
        Save the fitted model to disk.
        
        Args:
            path: Path to save directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        with open(path / "nb_model.pkl", "wb") as f:
            pickle.dump(self.nb_model, f)
        
        with open(path / "lr_model.pkl", "wb") as f:
            pickle.dump(self.lr_model, f)
        
        config = {
            "nb_weight": self.nb_weight,
            "lr_weight": self.lr_weight,
            "vectorizer_type": self.vectorizer_type,
            "feature_names": self._feature_names
        }
        with open(path / "config.pkl", "wb") as f:
            pickle.dump(config, f)
    
    def load(self, path: Path) -> "EnsembleClassifier":
        """
        Load a fitted model from disk.
        
        Args:
            path: Path to saved model directory.
            
        Returns:
            Self for method chaining.
        """
        path = Path(path)
        
        with open(path / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        
        with open(path / "nb_model.pkl", "rb") as f:
            self.nb_model = pickle.load(f)
        
        with open(path / "lr_model.pkl", "rb") as f:
            self.lr_model = pickle.load(f)
        
        with open(path / "config.pkl", "rb") as f:
            config = pickle.load(f)
            self.nb_weight = config["nb_weight"]
            self.lr_weight = config["lr_weight"]
            self.vectorizer_type = config["vectorizer_type"]
            self._feature_names = config["feature_names"]
        
        self._fitted = True
        return self
