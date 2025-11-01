"""
Correlation analysis between sentiment and price movements.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class CorrelationAnalyzer:
    """Analyzes correlation between sentiment and price movements."""

    def __init__(self):
        """Initialize correlation analyzer."""
        self.scaler = StandardScaler()

    def calculate_price_changes(
        self, price_df: pd.DataFrame, periods: List[int] = [1, 3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Calculate price changes over different periods.

        Args:
            price_df: DataFrame with price data (datetime index).
            periods: List of periods (hours) for calculating changes.

        Returns:
            DataFrame with price changes.
        """
        df = price_df.copy()

        for col in df.columns:
            for period in periods:
                # Price change
                df[f"{col}_change_{period}h"] = df[col].pct_change(periods=period)

                # Absolute change
                df[f"{col}_abs_change_{period}h"] = df[col].diff(periods=period)

                # Direction (1 for up, -1 for down, 0 for no change)
                df[f"{col}_direction_{period}h"] = np.sign(df[f"{col}_change_{period}h"])

        return df

    def calculate_sentiment_aggregates(
        self, sentiment_df: pd.DataFrame, time_column: str = "published_at", freq: str = "H"
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores by time period.

        Args:
            sentiment_df: DataFrame with sentiment scores.
            time_column: Column name for timestamps.
            freq: Frequency for aggregation (H=hourly, D=daily).

        Returns:
            DataFrame with aggregated sentiment.
        """
        df = sentiment_df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        df.set_index(time_column, inplace=True)

        # Aggregate sentiment
        agg_dict = {}

        if "sentiment_score" in df.columns:
            agg_dict["sentiment_score"] = ["mean", "std", "min", "max", "count"]

        if "vader_compound" in df.columns:
            agg_dict["vader_compound"] = ["mean", "std"]

        if "textblob_polarity" in df.columns:
            agg_dict["textblob_polarity"] = ["mean", "std"]

        agg_df = df.resample(freq).agg(agg_dict)

        # Flatten column names
        agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]

        return agg_df

    def pearson_correlation(
        self, sentiment_series: pd.Series, price_series: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate Pearson correlation coefficient.

        Args:
            sentiment_series: Sentiment scores.
            price_series: Price changes or levels.

        Returns:
            Tuple of (correlation coefficient, p-value).
        """
        # Remove NaN values
        combined = pd.DataFrame({"sentiment": sentiment_series, "price": price_series}).dropna()

        if len(combined) < 3:
            return 0.0, 1.0

        corr, pval = stats.pearsonr(combined["sentiment"], combined["price"])
        return corr, pval

    def spearman_correlation(
        self, sentiment_series: pd.Series, price_series: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate Spearman rank correlation coefficient.

        Args:
            sentiment_series: Sentiment scores.
            price_series: Price changes or levels.

        Returns:
            Tuple of (correlation coefficient, p-value).
        """
        # Remove NaN values
        combined = pd.DataFrame({"sentiment": sentiment_series, "price": price_series}).dropna()

        if len(combined) < 3:
            return 0.0, 1.0

        corr, pval = stats.spearmanr(combined["sentiment"], combined["price"])
        return corr, pval

    def lagged_correlation(
        self,
        sentiment_series: pd.Series,
        price_series: pd.Series,
        max_lag: int = 24,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        Calculate correlation with different time lags.

        Args:
            sentiment_series: Sentiment scores.
            price_series: Price changes or levels.
            max_lag: Maximum lag in periods to test.
            method: 'pearson' or 'spearman'.

        Returns:
            DataFrame with lag, correlation, and p-value.
        """
        results = []

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                sent = sentiment_series
                price = price_series
            elif lag > 0:
                # Sentiment leads price
                sent = sentiment_series[:-lag] if lag > 0 else sentiment_series
                price = price_series[lag:]
            else:
                # Price leads sentiment
                sent = sentiment_series[-lag:]
                price = price_series[:lag] if lag < 0 else price_series

            # Align indices
            combined = pd.DataFrame({"sentiment": sent, "price": price}).dropna()

            if len(combined) < 3:
                corr, pval = 0.0, 1.0
            else:
                if method == "pearson":
                    corr, pval = stats.pearsonr(combined["sentiment"], combined["price"])
                else:
                    corr, pval = stats.spearmanr(combined["sentiment"], combined["price"])

            results.append({"lag": lag, "correlation": corr, "p_value": pval})

        return pd.DataFrame(results)

    def analyze_sentiment_price_correlation(
        self, sentiment_df: pd.DataFrame, price_df: pd.DataFrame, asset: str = "BTC-USD"
    ) -> Dict:
        """
        Comprehensive correlation analysis between sentiment and prices.

        Args:
            sentiment_df: DataFrame with sentiment data and timestamps.
            price_df: DataFrame with price data (datetime index).
            asset: Asset symbol in price_df.

        Returns:
            Dictionary with correlation analysis results.
        """
        # Aggregate sentiment by hour
        sentiment_agg = self.calculate_sentiment_aggregates(sentiment_df)

        # Calculate price changes
        price_changes = self.calculate_price_changes(price_df)

        # Merge sentiment and price data
        merged = sentiment_agg.merge(
            price_changes, left_index=True, right_index=True, how="inner"
        )

        results = {"asset": asset, "samples": len(merged), "correlations": {}}

        # Calculate correlations for different time horizons
        if "sentiment_score_mean" in merged.columns:
            sentiment_col = "sentiment_score_mean"

            for period in [1, 3, 6, 12, 24]:
                price_change_col = f"{asset}_change_{period}h"

                if price_change_col in merged.columns:
                    # Pearson correlation
                    pearson_corr, pearson_pval = self.pearson_correlation(
                        merged[sentiment_col], merged[price_change_col]
                    )

                    # Spearman correlation
                    spearman_corr, spearman_pval = self.spearman_correlation(
                        merged[sentiment_col], merged[price_change_col]
                    )

                    results["correlations"][f"{period}h"] = {
                        "pearson": {"correlation": pearson_corr, "p_value": pearson_pval},
                        "spearman": {"correlation": spearman_corr, "p_value": spearman_pval},
                    }

        # Find lag with strongest correlation
        if "sentiment_score_mean" in merged.columns and f"{asset}_change_1h" in merged.columns:
            lagged = self.lagged_correlation(
                merged["sentiment_score_mean"], merged[f"{asset}_change_1h"], max_lag=12
            )

            # Find best lag
            best_lag_idx = lagged["correlation"].abs().idxmax()
            best_lag = lagged.loc[best_lag_idx]

            results["best_lag"] = {
                "lag": int(best_lag["lag"]),
                "correlation": float(best_lag["correlation"]),
                "p_value": float(best_lag["p_value"]),
            }

        return results

    def calculate_sentiment_impact_score(
        self, sentiment_df: pd.DataFrame, price_df: pd.DataFrame, window: int = 24
    ) -> pd.DataFrame:
        """
        Calculate sentiment impact score based on correlation strength.

        Args:
            sentiment_df: DataFrame with sentiment data.
            price_df: DataFrame with price data.
            window: Rolling window size (hours).

        Returns:
            DataFrame with impact scores.
        """
        # Aggregate sentiment
        sentiment_agg = self.calculate_sentiment_aggregates(sentiment_df)

        # Calculate price changes
        price_changes = self.calculate_price_changes(price_df, periods=[1])

        # Merge
        merged = sentiment_agg.merge(
            price_changes, left_index=True, right_index=True, how="inner"
        )

        # Calculate rolling correlation
        if "sentiment_score_mean" in merged.columns:
            for col in price_changes.columns:
                if "_change_1h" in col:
                    merged[f"rolling_corr_{col}"] = (
                        merged["sentiment_score_mean"]
                        .rolling(window)
                        .corr(merged[col])
                    )

        return merged

    def identify_sentiment_events(
        self, sentiment_df: pd.DataFrame, threshold: float = 0.5, min_articles: int = 5
    ) -> pd.DataFrame:
        """
        Identify significant sentiment events.

        Args:
            sentiment_df: DataFrame with sentiment data.
            threshold: Minimum absolute sentiment score to consider.
            min_articles: Minimum number of articles for an event.

        Returns:
            DataFrame with sentiment events.
        """
        # Aggregate by hour
        agg_df = self.calculate_sentiment_aggregates(sentiment_df)

        # Filter significant events
        events = agg_df[
            (abs(agg_df["sentiment_score_mean"]) >= threshold)
            & (agg_df["sentiment_score_count"] >= min_articles)
        ].copy()

        # Add event metadata
        events["event_strength"] = abs(events["sentiment_score_mean"])
        events["event_direction"] = np.sign(events["sentiment_score_mean"])

        return events.sort_values("event_strength", ascending=False)


class PredictiveAnalyzer:
    """Analyze predictive power of sentiment on price movements."""

    def __init__(self):
        """Initialize predictive analyzer."""
        pass

    def calculate_directional_accuracy(
        self, sentiment_df: pd.DataFrame, price_df: pd.DataFrame, asset: str = "BTC-USD", lag: int = 1
    ) -> Dict:
        """
        Calculate how accurately sentiment predicts price direction.

        Args:
            sentiment_df: DataFrame with sentiment data.
            price_df: DataFrame with price data.
            asset: Asset symbol.
            lag: Lag in hours (sentiment leads price).

        Returns:
            Dictionary with accuracy metrics.
        """
        # Aggregate sentiment
        sentiment_agg = sentiment_df.copy()
        sentiment_agg["published_at"] = pd.to_datetime(sentiment_agg["published_at"])
        sentiment_agg = sentiment_agg.set_index("published_at").resample("H").mean()

        # Calculate price changes
        price_change = price_df[asset].pct_change(periods=lag)

        # Merge with lag
        if lag > 0:
            # Shift price to the future
            price_change_shifted = price_change.shift(-lag)
        else:
            price_change_shifted = price_change

        merged = pd.DataFrame(
            {"sentiment": sentiment_agg.get("sentiment_score", 0), "price_change": price_change_shifted}
        ).dropna()

        if len(merged) < 10:
            return {"accuracy": 0.0, "samples": len(merged)}

        # Determine directions
        sentiment_direction = np.sign(merged["sentiment"])
        price_direction = np.sign(merged["price_change"])

        # Calculate accuracy
        correct_predictions = (sentiment_direction == price_direction).sum()
        accuracy = correct_predictions / len(merged)

        # Separate positive and negative sentiment accuracy
        positive_sentiment = merged[merged["sentiment"] > 0]
        negative_sentiment = merged[merged["sentiment"] < 0]

        pos_accuracy = (
            (np.sign(positive_sentiment["sentiment"]) == np.sign(positive_sentiment["price_change"])).mean()
            if len(positive_sentiment) > 0
            else 0.0
        )

        neg_accuracy = (
            (np.sign(negative_sentiment["sentiment"]) == np.sign(negative_sentiment["price_change"])).mean()
            if len(negative_sentiment) > 0
            else 0.0
        )

        return {
            "overall_accuracy": accuracy,
            "positive_sentiment_accuracy": pos_accuracy,
            "negative_sentiment_accuracy": neg_accuracy,
            "total_samples": len(merged),
            "positive_samples": len(positive_sentiment),
            "negative_samples": len(negative_sentiment),
        }


# Example usage
if __name__ == "__main__":
    print("Testing CorrelationAnalyzer...")

    # Create sample data
    dates = pd.date_range("2025-11-01", periods=100, freq="H")

    # Sample sentiment data
    np.random.seed(42)
    sentiment_data = {
        "published_at": dates,
        "sentiment_score": np.random.randn(100) * 0.3,
        "vader_compound": np.random.randn(100) * 0.4,
    }
    sentiment_df = pd.DataFrame(sentiment_data)

    # Sample price data (correlated with sentiment)
    price_changes = sentiment_data["sentiment_score"] * 0.02 + np.random.randn(100) * 0.01
    prices = 50000 * (1 + pd.Series(price_changes).cumsum())

    price_df = pd.DataFrame({"BTC-USD": prices}, index=dates)

    # Analyze correlation
    analyzer = CorrelationAnalyzer()
    results = analyzer.analyze_sentiment_price_correlation(sentiment_df, price_df)

    print(f"\nCorrelation Analysis Results:")
    print(f"Samples: {results['samples']}")
    print(f"\nCorrelations by time horizon:")
    for period, corr in results.get("correlations", {}).items():
        print(f"{period}: Pearson={corr['pearson']['correlation']:.3f}, p={corr['pearson']['p_value']:.3f}")

    if "best_lag" in results:
        print(f"\nBest lag: {results['best_lag']['lag']} hours")
        print(f"Correlation at best lag: {results['best_lag']['correlation']:.3f}")
