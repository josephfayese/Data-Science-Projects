"""
Sentiment analysis using multiple NLP models.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re


class VaderAnalyzer:
    """VADER sentiment analysis - optimized for social media."""

    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using VADER.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with sentiment scores (neg, neu, pos, compound).
        """
        if not text or not isinstance(text, str):
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        scores = self.analyzer.polarity_scores(text)
        return scores

    def get_sentiment_label(self, compound_score: float) -> str:
        """
        Convert compound score to sentiment label.

        Args:
            compound_score: VADER compound score (-1 to 1).

        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'.
        """
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"


class TextBlobAnalyzer:
    """TextBlob sentiment analysis."""

    def __init__(self):
        """Initialize TextBlob analyzer."""
        pass

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using TextBlob.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with polarity and subjectivity scores.
        """
        if not text or not isinstance(text, str):
            return {"polarity": 0.0, "subjectivity": 0.0}

        blob = TextBlob(text)
        return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}

    def get_sentiment_label(self, polarity: float) -> str:
        """
        Convert polarity score to sentiment label.

        Args:
            polarity: TextBlob polarity score (-1 to 1).

        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'.
        """
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"


class FinancialKeywordAnalyzer:
    """Financial-specific keyword-based sentiment analysis."""

    def __init__(self):
        """Initialize financial keyword analyzer."""
        self.positive_keywords = [
            "surge",
            "rally",
            "gain",
            "rise",
            "increase",
            "bull",
            "bullish",
            "growth",
            "profit",
            "boost",
            "strong",
            "recovery",
            "positive",
            "upward",
            "stabilize",
            "improve",
            "strengthen",
        ]

        self.negative_keywords = [
            "crash",
            "plunge",
            "fall",
            "decline",
            "decrease",
            "bear",
            "bearish",
            "loss",
            "weak",
            "crisis",
            "negative",
            "downward",
            "volatility",
            "unstable",
            "ban",
            "restrict",
            "regulation",
        ]

        self.neutral_keywords = [
            "stable",
            "unchanged",
            "flat",
            "steady",
            "maintain",
            "hold",
        ]

    def analyze(self, text: str) -> Dict[str, Union[float, int]]:
        """
        Analyze sentiment based on financial keywords.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with keyword counts and sentiment score.
        """
        if not text or not isinstance(text, str):
            return {
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "sentiment_score": 0.0,
            }

        text_lower = text.lower()

        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        neutral_count = sum(1 for keyword in self.neutral_keywords if keyword in text_lower)

        total_keywords = positive_count + negative_count + neutral_count

        if total_keywords > 0:
            sentiment_score = (positive_count - negative_count) / total_keywords
        else:
            sentiment_score = 0.0

        return {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "sentiment_score": sentiment_score,
        }


class EnsembleAnalyzer:
    """Ensemble sentiment analyzer combining multiple methods."""

    def __init__(self):
        """Initialize ensemble analyzer with multiple models."""
        self.vader = VaderAnalyzer()
        self.textblob = TextBlobAnalyzer()
        self.financial = FinancialKeywordAnalyzer()

    def analyze(self, text: str) -> Dict[str, Union[float, str, Dict]]:
        """
        Analyze sentiment using ensemble of methods.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with all sentiment scores and ensemble results.
        """
        if not text or not isinstance(text, str):
            return {
                "text": text,
                "vader": {"compound": 0.0},
                "textblob": {"polarity": 0.0},
                "financial": {"sentiment_score": 0.0},
                "ensemble_score": 0.0,
                "ensemble_label": "neutral",
            }

        # Get scores from each analyzer
        vader_scores = self.vader.analyze(text)
        textblob_scores = self.textblob.analyze(text)
        financial_scores = self.financial.analyze(text)

        # Calculate ensemble score (weighted average)
        ensemble_score = (
            vader_scores["compound"] * 0.4
            + textblob_scores["polarity"] * 0.3
            + financial_scores["sentiment_score"] * 0.3
        )

        # Determine ensemble label
        if ensemble_score >= 0.15:
            ensemble_label = "positive"
        elif ensemble_score <= -0.15:
            ensemble_label = "negative"
        else:
            ensemble_label = "neutral"

        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "vader": vader_scores,
            "textblob": textblob_scores,
            "financial": financial_scores,
            "ensemble_score": ensemble_score,
            "ensemble_label": ensemble_label,
        }

    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            DataFrame with sentiment analysis results.
        """
        results = []

        for text in texts:
            analysis = self.analyze(text)

            # Flatten the results
            results.append(
                {
                    "text": analysis["text"],
                    "vader_compound": analysis["vader"]["compound"],
                    "vader_pos": analysis["vader"].get("pos", 0),
                    "vader_neg": analysis["vader"].get("neg", 0),
                    "vader_neu": analysis["vader"].get("neu", 0),
                    "textblob_polarity": analysis["textblob"]["polarity"],
                    "textblob_subjectivity": analysis["textblob"]["subjectivity"],
                    "financial_score": analysis["financial"]["sentiment_score"],
                    "financial_pos_count": analysis["financial"]["positive_count"],
                    "financial_neg_count": analysis["financial"]["negative_count"],
                    "ensemble_score": analysis["ensemble_score"],
                    "ensemble_label": analysis["ensemble_label"],
                }
            )

        return pd.DataFrame(results)


class ArticleSentimentAnalyzer:
    """Specialized analyzer for news articles and social media posts."""

    def __init__(self):
        """Initialize article sentiment analyzer."""
        self.ensemble = EnsembleAnalyzer()

    def analyze_article(self, article: Dict) -> Dict:
        """
        Analyze sentiment of a news article.

        Args:
            article: Article dictionary with 'title', 'description', 'content'.

        Returns:
            Article with added sentiment analysis.
        """
        # Combine title and description/content for analysis
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")

        # Title has more weight
        combined_text = f"{title}. {title}. {description} {content}"

        # Analyze sentiment
        sentiment = self.ensemble.analyze(combined_text)

        # Add sentiment to article
        article_with_sentiment = article.copy()
        article_with_sentiment.update(
            {
                "sentiment_score": sentiment["ensemble_score"],
                "sentiment_label": sentiment["ensemble_label"],
                "vader_compound": sentiment["vader"]["compound"],
                "textblob_polarity": sentiment["textblob"]["polarity"],
                "financial_score": sentiment["financial"]["sentiment_score"],
            }
        )

        return article_with_sentiment

    def analyze_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for multiple articles.

        Args:
            articles: List of article dictionaries.

        Returns:
            List of articles with sentiment analysis.
        """
        return [self.analyze_article(article) for article in articles]

    def analyze_tweets(self, tweets: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for tweets.

        Args:
            tweets: List of tweet dictionaries.

        Returns:
            List of tweets with sentiment analysis.
        """
        results = []

        for tweet in tweets:
            text = tweet.get("text", "")

            if text:
                sentiment = self.ensemble.analyze(text)

                tweet_with_sentiment = tweet.copy()
                tweet_with_sentiment.update(
                    {
                        "sentiment_score": sentiment["ensemble_score"],
                        "sentiment_label": sentiment["ensemble_label"],
                        "vader_compound": sentiment["vader"]["compound"],
                        "textblob_polarity": sentiment["textblob"]["polarity"],
                    }
                )

                results.append(tweet_with_sentiment)

        return results


# Example usage
if __name__ == "__main__":
    # Test analyzers
    test_texts = [
        "Bitcoin surges as Nigeria announces positive crypto regulations",
        "Central Bank restricts forex trading, Naira plunges",
        "Bitcoin price remains stable amid policy uncertainty",
        "CBN announces new policy to strengthen the Naira against USD",
    ]

    print("Testing EnsembleAnalyzer...")
    ensemble = EnsembleAnalyzer()

    for text in test_texts:
        print(f"\nText: {text}")
        result = ensemble.analyze(text)
        print(f"Ensemble Score: {result['ensemble_score']:.3f}")
        print(f"Ensemble Label: {result['ensemble_label']}")
        print(f"VADER Compound: {result['vader']['compound']:.3f}")
        print(f"TextBlob Polarity: {result['textblob']['polarity']:.3f}")
        print(f"Financial Score: {result['financial']['sentiment_score']:.3f}")

    print("\n" + "=" * 80)
    print("Testing batch analysis...")
    df = ensemble.analyze_batch(test_texts)
    print(df[["text", "ensemble_score", "ensemble_label"]])
