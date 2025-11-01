"""
Data preprocessing and cleaning utilities.
"""

import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Union
import pandas as pd
from pathlib import Path


class TextPreprocessor:
    """Preprocessor for text data cleaning."""

    def __init__(self):
        """Initialize text preprocessor."""
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
        self.mention_pattern = re.compile(r"@[A-Za-z0-9_]+")
        self.hashtag_pattern = re.compile(r"#")
        self.special_chars_pattern = re.compile(r"[^A-Za-z0-9\s.,!?-]")
        self.whitespace_pattern = re.compile(r"\s+")

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.url_pattern.sub("", text)

    def remove_mentions(self, text: str) -> str:
        """Remove @ mentions from text."""
        return self.mention_pattern.sub("", text)

    def remove_hashtags(self, text: str, keep_text: bool = True) -> str:
        """
        Remove or process hashtags.

        Args:
            text: Input text.
            keep_text: If True, keep hashtag text without #.

        Returns:
            Processed text.
        """
        if keep_text:
            return self.hashtag_pattern.sub("", text)
        else:
            return re.sub(r"#\w+", "", text)

    def remove_special_chars(self, text: str) -> str:
        """Remove special characters, keep basic punctuation."""
        return self.special_chars_pattern.sub(" ", text)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace to single spaces."""
        return self.whitespace_pattern.sub(" ", text).strip()

    def clean_text(
        self,
        text: str,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        lowercase: bool = False,
    ) -> str:
        """
        Clean text with multiple preprocessing steps.

        Args:
            text: Input text.
            remove_urls: Remove URLs.
            remove_mentions: Remove @ mentions.
            remove_hashtags: Remove hashtags.
            lowercase: Convert to lowercase.

        Returns:
            Cleaned text.
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned = text

        if remove_urls:
            cleaned = self.remove_urls(cleaned)

        if remove_mentions:
            cleaned = self.remove_mentions(cleaned)

        if remove_hashtags:
            cleaned = self.remove_hashtags(cleaned, keep_text=True)

        # Remove special characters but keep basic punctuation
        cleaned = self.remove_special_chars(cleaned)

        # Normalize whitespace
        cleaned = self.normalize_whitespace(cleaned)

        if lowercase:
            cleaned = cleaned.lower()

        return cleaned


class DataFramePreprocessor:
    """Preprocessor for DataFrame operations."""

    def __init__(self):
        """Initialize DataFrame preprocessor."""
        self.text_preprocessor = TextPreprocessor()

    def load_json_data(self, filepath: Union[str, Path]) -> List[Dict]:
        """
        Load data from JSON file.

        Args:
            filepath: Path to JSON file.

        Returns:
            List of data dictionaries.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def articles_to_dataframe(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Convert articles list to DataFrame with preprocessing.

        Args:
            articles: List of article dictionaries.

        Returns:
            DataFrame with cleaned articles.
        """
        df = pd.DataFrame(articles)

        # Clean text fields
        if "title" in df.columns:
            df["title_clean"] = df["title"].apply(
                lambda x: self.text_preprocessor.clean_text(x) if pd.notna(x) else ""
            )

        if "description" in df.columns:
            df["description_clean"] = df["description"].apply(
                lambda x: self.text_preprocessor.clean_text(x) if pd.notna(x) else ""
            )

        if "content" in df.columns:
            df["content_clean"] = df["content"].apply(
                lambda x: self.text_preprocessor.clean_text(x) if pd.notna(x) else ""
            )

        # Parse dates
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

        if "collected_at" in df.columns:
            df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce")

        return df

    def tweets_to_dataframe(self, tweets: List[Dict]) -> pd.DataFrame:
        """
        Convert tweets list to DataFrame with preprocessing.

        Args:
            tweets: List of tweet dictionaries.

        Returns:
            DataFrame with cleaned tweets.
        """
        df = pd.DataFrame(tweets)

        # Clean tweet text
        if "text" in df.columns:
            df["text_clean"] = df["text"].apply(
                lambda x: self.text_preprocessor.clean_text(
                    x, remove_urls=True, remove_mentions=False, remove_hashtags=False
                )
                if pd.notna(x)
                else ""
            )

        # Parse dates
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

        if "collected_at" in df.columns:
            df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce")

        return df

    def merge_sentiment_with_prices(
        self, sentiment_df: pd.DataFrame, price_df: pd.DataFrame, time_column: str = "published_at"
    ) -> pd.DataFrame:
        """
        Merge sentiment data with price data based on timestamps.

        Args:
            sentiment_df: DataFrame with sentiment data and timestamps.
            price_df: DataFrame with price data and timestamps.
            time_column: Column name for timestamps in sentiment_df.

        Returns:
            Merged DataFrame.
        """
        # Ensure datetime types
        sentiment_df[time_column] = pd.to_datetime(sentiment_df[time_column])

        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.to_datetime(price_df.index)

        # Resample price data to hourly
        price_hourly = price_df.resample("H").last().ffill()

        # Round sentiment timestamps to nearest hour
        sentiment_df["hour"] = sentiment_df[time_column].dt.floor("H")

        # Merge
        merged = sentiment_df.merge(
            price_hourly, left_on="hour", right_index=True, how="left", suffixes=("", "_price")
        )

        return merged

    def aggregate_sentiment_by_time(
        self,
        df: pd.DataFrame,
        time_column: str = "published_at",
        sentiment_column: str = "sentiment_score",
        freq: str = "H",
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores by time period.

        Args:
            df: DataFrame with sentiment data.
            time_column: Column name for timestamps.
            sentiment_column: Column name for sentiment scores.
            freq: Frequency for aggregation (H=hourly, D=daily).

        Returns:
            DataFrame with aggregated sentiment.
        """
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])

        # Set time as index
        df.set_index(time_column, inplace=True)

        # Aggregate
        agg_df = df[sentiment_column].resample(freq).agg(["mean", "std", "count"])

        agg_df.columns = [f"sentiment_{col}" for col in agg_df.columns]

        return agg_df

    def filter_by_keywords(self, df: pd.DataFrame, keywords: List[str], text_column: str = "text") -> pd.DataFrame:
        """
        Filter DataFrame by keywords in text column.

        Args:
            df: Input DataFrame.
            keywords: List of keywords to filter by.
            text_column: Column name containing text.

        Returns:
            Filtered DataFrame.
        """
        if text_column not in df.columns:
            return df

        # Create regex pattern for keywords
        pattern = "|".join([re.escape(keyword) for keyword in keywords])

        # Filter
        mask = df[text_column].str.contains(pattern, case=False, na=False)

        return df[mask]

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            df: Input DataFrame.
            subset: Columns to consider for duplicates.

        Returns:
            DataFrame without duplicates.
        """
        return df.drop_duplicates(subset=subset, keep="first")


class FeatureEngineer:
    """Feature engineering for sentiment analysis."""

    def __init__(self):
        """Initialize feature engineer."""
        pass

    def add_time_features(self, df: pd.DataFrame, time_column: str = "published_at") -> pd.DataFrame:
        """
        Add time-based features.

        Args:
            df: Input DataFrame.
            time_column: Column name for timestamps.

        Returns:
            DataFrame with added time features.
        """
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])

        df["hour"] = df[time_column].dt.hour
        df["day_of_week"] = df[time_column].dt.dayofweek
        df["day_of_month"] = df[time_column].dt.day
        df["month"] = df[time_column].dt.month
        df["year"] = df[time_column].dt.year
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Trading hours (approximate - can be customized)
        df["is_trading_hours"] = df["hour"].between(9, 17).astype(int)

        return df

    def add_text_features(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Add text-based features.

        Args:
            df: Input DataFrame.
            text_column: Column name containing text.

        Returns:
            DataFrame with added text features.
        """
        df = df.copy()

        if text_column in df.columns:
            df["text_length"] = df[text_column].str.len()
            df["word_count"] = df[text_column].str.split().str.len()
            df["avg_word_length"] = df["text_length"] / (df["word_count"] + 1)
            df["exclamation_count"] = df[text_column].str.count("!")
            df["question_count"] = df[text_column].str.count(r"\?")

        return df

    def add_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engagement features for social media data.

        Args:
            df: Input DataFrame with social media metrics.

        Returns:
            DataFrame with added engagement features.
        """
        df = df.copy()

        # Calculate engagement rate for tweets
        if all(col in df.columns for col in ["retweet_count", "like_count", "author_followers"]):
            df["engagement_rate"] = (df["retweet_count"] + df["like_count"]) / (
                df["author_followers"] + 1
            )

        # Calculate total engagement
        if all(col in df.columns for col in ["retweet_count", "reply_count", "like_count"]):
            df["total_engagement"] = (
                df["retweet_count"] + df["reply_count"] + df["like_count"]
            )

        return df


# Example usage
if __name__ == "__main__":
    # Test preprocessors
    print("Testing TextPreprocessor...")
    preprocessor = TextPreprocessor()

    test_text = "Check out this amazing article! https://example.com @user #bitcoin #crypto"
    cleaned = preprocessor.clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")

    print("\nTesting DataFramePreprocessor...")
    df_preprocessor = DataFramePreprocessor()

    # Create sample data
    sample_articles = [
        {
            "title": "Bitcoin Surges on Policy News",
            "description": "BTC rises after announcement",
            "published_at": "2025-11-01T10:00:00Z",
        },
        {
            "title": "Naira Falls Against Dollar",
            "description": "USD/NGN hits new high",
            "published_at": "2025-11-01T11:00:00Z",
        },
    ]

    df = df_preprocessor.articles_to_dataframe(sample_articles)
    print(f"\nArticles DataFrame shape: {df.shape}")
    print(df[["title", "title_clean"]].head())
