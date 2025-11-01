"""
Visualization utilities for sentiment analysis and price data.
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


class SentimentVisualizer:
    """Visualizations for sentiment analysis."""

    def __init__(self, style: str = "seaborn"):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style to use.
        """
        self.style = style
        if style != "default":
            plt.style.use(style)

    def plot_sentiment_distribution(
        self, sentiment_df: pd.DataFrame, sentiment_col: str = "sentiment_score", save_path: Optional[str] = None
    ):
        """
        Plot distribution of sentiment scores.

        Args:
            sentiment_df: DataFrame with sentiment data.
            sentiment_col: Column name for sentiment scores.
            save_path: Path to save figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(sentiment_df[sentiment_col].dropna(), bins=50, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Sentiment Score")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Sentiment Score Distribution")
        axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5)

        # Box plot by label
        if "sentiment_label" in sentiment_df.columns:
            sentiment_df.boxplot(column=sentiment_col, by="sentiment_label", ax=axes[1])
            axes[1].set_xlabel("Sentiment Label")
            axes[1].set_ylabel("Sentiment Score")
            axes[1].set_title("Sentiment Score by Label")
            plt.suptitle("")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_sentiment_timeline(
        self,
        sentiment_df: pd.DataFrame,
        time_col: str = "published_at",
        sentiment_col: str = "sentiment_score",
        freq: str = "D",
        save_path: Optional[str] = None,
    ):
        """
        Plot sentiment over time.

        Args:
            sentiment_df: DataFrame with sentiment data.
            time_col: Column name for timestamps.
            sentiment_col: Column name for sentiment scores.
            freq: Frequency for aggregation.
            save_path: Path to save figure.
        """
        df = sentiment_df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)

        # Aggregate sentiment
        agg_df = df[sentiment_col].resample(freq).agg(["mean", "std", "count"])

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot mean sentiment with error bands
        ax.plot(agg_df.index, agg_df["mean"], label="Mean Sentiment", linewidth=2)
        ax.fill_between(
            agg_df.index,
            agg_df["mean"] - agg_df["std"],
            agg_df["mean"] + agg_df["std"],
            alpha=0.3,
            label="±1 Std Dev",
        )

        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sentiment Score")
        ax.set_title(f"Sentiment Over Time (Aggregated by {freq})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_sentiment_heatmap(
        self,
        sentiment_df: pd.DataFrame,
        time_col: str = "published_at",
        sentiment_col: str = "sentiment_score",
        save_path: Optional[str] = None,
    ):
        """
        Plot sentiment heatmap by hour and day of week.

        Args:
            sentiment_df: DataFrame with sentiment data.
            time_col: Column name for timestamps.
            sentiment_col: Column name for sentiment scores.
            save_path: Path to save figure.
        """
        df = sentiment_df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

        # Add hour and day of week
        df["hour"] = df[time_col].dt.hour
        df["day_of_week"] = df[time_col].dt.day_name()

        # Create pivot table
        pivot = df.pivot_table(
            values=sentiment_col, index="day_of_week", columns="hour", aggfunc="mean"
        )

        # Reorder days
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot.reindex([day for day in day_order if day in pivot.index])

        # Plot heatmap
        plt.figure(figsize=(14, 6))
        sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=False, fmt=".2f", cbar_kws={"label": "Mean Sentiment"})
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week")
        plt.title("Sentiment Heatmap by Hour and Day")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


class PriceVisualizer:
    """Visualizations for price data."""

    def __init__(self):
        """Initialize price visualizer."""
        pass

    def plot_price_timeline(
        self, price_df: pd.DataFrame, assets: Optional[List[str]] = None, save_path: Optional[str] = None
    ):
        """
        Plot price timeline for multiple assets.

        Args:
            price_df: DataFrame with price data (datetime index).
            assets: List of asset columns to plot.
            save_path: Path to save figure.
        """
        if assets is None:
            assets = price_df.columns.tolist()

        fig, axes = plt.subplots(len(assets), 1, figsize=(14, 4 * len(assets)))

        if len(assets) == 1:
            axes = [axes]

        for i, asset in enumerate(assets):
            if asset in price_df.columns:
                axes[i].plot(price_df.index, price_df[asset], linewidth=2)
                axes[i].set_xlabel("Date")
                axes[i].set_ylabel("Price")
                axes[i].set_title(f"{asset} Price")
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


class CorrelationVisualizer:
    """Visualizations for correlation analysis."""

    def __init__(self):
        """Initialize correlation visualizer."""
        pass

    def plot_sentiment_price_correlation(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        sentiment_col: str = "sentiment_score",
        price_col: str = "BTC-USD",
        save_path: Optional[str] = None,
    ):
        """
        Plot sentiment vs price changes with correlation.

        Args:
            sentiment_df: DataFrame with sentiment data.
            price_df: DataFrame with price data.
            sentiment_col: Column name for sentiment.
            price_col: Column name for price.
            save_path: Path to save figure.
        """
        # Merge data
        sent_agg = sentiment_df.copy()
        sent_agg["published_at"] = pd.to_datetime(sent_agg["published_at"])
        sent_agg = sent_agg.set_index("published_at").resample("H").mean()

        price_change = price_df[price_col].pct_change()

        merged = pd.DataFrame({"sentiment": sent_agg[sentiment_col], "price_change": price_change}).dropna()

        # Calculate correlation
        corr = merged["sentiment"].corr(merged["price_change"])

        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(merged["sentiment"], merged["price_change"], alpha=0.5)

        # Add trend line
        z = np.polyfit(merged["sentiment"], merged["price_change"], 1)
        p = np.poly1d(z)
        plt.plot(merged["sentiment"], p(merged["sentiment"]), "r--", alpha=0.8, linewidth=2)

        plt.xlabel("Sentiment Score")
        plt.ylabel("Price Change (%)")
        plt.title(f"Sentiment vs Price Change\nCorrelation: {corr:.3f}")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_lagged_correlation(self, lagged_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot correlation at different lags.

        Args:
            lagged_df: DataFrame with lag and correlation columns.
            save_path: Path to save figure.
        """
        plt.figure(figsize=(12, 6))

        # Plot correlation vs lag
        plt.plot(lagged_df["lag"], lagged_df["correlation"], marker="o", linewidth=2)

        # Add significance threshold
        plt.axhline(y=0.3, color="green", linestyle="--", alpha=0.5, label="Strong Positive")
        plt.axhline(y=-0.3, color="red", linestyle="--", alpha=0.5, label="Strong Negative")
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        plt.xlabel("Lag (hours)")
        plt.ylabel("Correlation Coefficient")
        plt.title("Sentiment-Price Correlation at Different Lags")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


class InteractiveDashboard:
    """Interactive visualizations using Plotly."""

    def __init__(self):
        """Initialize interactive dashboard."""
        pass

    def create_sentiment_price_dashboard(
        self, sentiment_df: pd.DataFrame, price_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive dashboard with sentiment and price.

        Args:
            sentiment_df: DataFrame with sentiment data.
            price_df: DataFrame with price data.
            save_path: Path to save HTML file.

        Returns:
            Plotly figure object.
        """
        # Aggregate sentiment
        sent_agg = sentiment_df.copy()
        sent_agg["published_at"] = pd.to_datetime(sent_agg["published_at"])
        sent_agg = sent_agg.set_index("published_at").resample("H").mean()

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Bitcoin Price", "Sentiment Score"),
            row_heights=[0.6, 0.4],
        )

        # Add price trace
        for col in price_df.columns:
            fig.add_trace(
                go.Scatter(x=price_df.index, y=price_df[col], name=col, mode="lines"), row=1, col=1
            )

        # Add sentiment trace
        if "sentiment_score" in sent_agg.columns:
            fig.add_trace(
                go.Scatter(
                    x=sent_agg.index,
                    y=sent_agg["sentiment_score"],
                    name="Sentiment",
                    mode="lines",
                    line=dict(color="green"),
                ),
                row=2,
                col=1,
            )

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)

        fig.update_layout(
            title="Sentiment and Price Analysis Dashboard",
            hovermode="x unified",
            height=800,
            showlegend=True,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_correlation_heatmap(self, corr_matrix: pd.DataFrame, save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive correlation heatmap.

        Args:
            corr_matrix: Correlation matrix.
            save_path: Path to save HTML file.

        Returns:
            Plotly figure object.
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="RdBu",
                zmid=0,
                text=corr_matrix.values,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
            )
        )

        fig.update_layout(title="Correlation Heatmap", xaxis_title="Variables", yaxis_title="Variables", height=600)

        if save_path:
            fig.write_html(save_path)

        return fig


# Example usage
if __name__ == "__main__":
    print("Testing visualizations...")

    # Create sample data
    dates = pd.date_range("2025-11-01", periods=100, freq="H")
    np.random.seed(42)

    sentiment_df = pd.DataFrame(
        {
            "published_at": dates,
            "sentiment_score": np.random.randn(100) * 0.3,
            "sentiment_label": np.random.choice(["positive", "negative", "neutral"], 100),
        }
    )

    price_df = pd.DataFrame(
        {"BTC-USD": 50000 + np.cumsum(np.random.randn(100) * 100), "USDNGN": 1600 + np.cumsum(np.random.randn(100))},
        index=dates,
    )

    # Test visualizers
    print("Creating sentiment visualizations...")
    sent_viz = SentimentVisualizer()
    sent_viz.plot_sentiment_distribution(sentiment_df)
    sent_viz.plot_sentiment_timeline(sentiment_df)

    print("Creating price visualizations...")
    price_viz = PriceVisualizer()
    price_viz.plot_price_timeline(price_df)

    print("Creating correlation visualizations...")
    corr_viz = CorrelationVisualizer()
    corr_viz.plot_sentiment_price_correlation(sentiment_df, price_df)

    print("Visualizations created successfully!")
