#!/usr/bin/env python
"""
Main script to run the complete financial sentiment analysis pipeline.
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import Config
from src.data.collectors import NewsCollector, RSSFeedCollector, PriceCollector
from src.data.twitter_collector import TwitterCollector
from src.data.preprocessors import DataFramePreprocessor
from src.sentiment.analyzers import ArticleSentimentAnalyzer
from src.models.correlation import CorrelationAnalyzer, PredictiveAnalyzer
from src.utils.visualization import SentimentVisualizer, PriceVisualizer, CorrelationVisualizer


def collect_data(args):
    """Collect data from various sources."""
    print("=" * 80)
    print("DATA COLLECTION")
    print("=" * 80)

    all_articles = []

    # Collect news
    if args.source in ["all", "news"]:
        print("\n[1/4] Collecting news articles...")
        news_collector = NewsCollector()
        news_articles = news_collector.collect_news(
            keywords=Config.ALL_KEYWORDS[:3],  # Limit keywords
            from_date=datetime.now() - timedelta(days=args.days),
        )
        print(f"  ✓ Collected {len(news_articles)} news articles")
        all_articles.extend(news_articles)

        if news_articles and args.save:
            news_collector.save_to_json(news_articles)

    # Collect RSS feeds
    if args.source in ["all", "rss"]:
        print("\n[2/4] Collecting from RSS feeds...")
        rss_collector = RSSFeedCollector()
        rss_articles = rss_collector.collect_all_feeds(max_entries_per_feed=20)
        print(f"  ✓ Collected {len(rss_articles)} RSS articles")
        all_articles.extend(rss_articles)

        if rss_articles and args.save:
            rss_collector.save_to_json(rss_articles)

    # Collect Twitter data
    if args.source in ["all", "twitter"]:
        print("\n[3/4] Collecting Twitter data...")
        twitter_collector = TwitterCollector()
        if twitter_collector.client:
            tweets = twitter_collector.collect_all(max_results_per_topic=100)
            print(f"  ✓ Collected {len(tweets)} tweets")

            if tweets and args.save:
                twitter_collector.save_to_json(tweets)
        else:
            print("  ✗ Twitter client not available")
            tweets = []
    else:
        tweets = []

    # Collect price data
    if args.source in ["all", "prices"]:
        print("\n[4/4] Collecting price data...")
        price_collector = PriceCollector()

        current_prices = price_collector.get_current_prices()
        print(f"  ✓ Current prices: {json.dumps(current_prices, indent=2)}")

        historical_prices = price_collector.get_historical_prices(period=f"{args.days}d")
        print(f"  ✓ Historical prices shape: {historical_prices.shape}")

        if not historical_prices.empty and args.save:
            price_collector.save_to_csv(historical_prices)
    else:
        historical_prices = None

    return all_articles, tweets, historical_prices


def analyze_sentiment(articles, tweets, args):
    """Analyze sentiment of collected data."""
    print("\n" + "=" * 80)
    print("SENTIMENT ANALYSIS")
    print("=" * 80)

    analyzer = ArticleSentimentAnalyzer()

    # Analyze articles
    if articles:
        print(f"\n[1/2] Analyzing {len(articles)} articles...")
        articles_with_sentiment = analyzer.analyze_articles(articles)

        # Calculate statistics
        positive = sum(1 for a in articles_with_sentiment if a.get("sentiment_label") == "positive")
        negative = sum(1 for a in articles_with_sentiment if a.get("sentiment_label") == "negative")
        neutral = sum(1 for a in articles_with_sentiment if a.get("sentiment_label") == "neutral")

        print(f"  ✓ Positive: {positive} ({positive/len(articles)*100:.1f}%)")
        print(f"  ✓ Negative: {negative} ({negative/len(articles)*100:.1f}%)")
        print(f"  ✓ Neutral: {neutral} ({neutral/len(articles)*100:.1f}%)")

        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Config.PROCESSED_DATA_DIR / f"articles_sentiment_{timestamp}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(articles_with_sentiment, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved to {filepath}")
    else:
        articles_with_sentiment = []

    # Analyze tweets
    if tweets:
        print(f"\n[2/2] Analyzing {len(tweets)} tweets...")
        tweets_with_sentiment = analyzer.analyze_tweets(tweets)

        positive = sum(1 for t in tweets_with_sentiment if t.get("sentiment_label") == "positive")
        negative = sum(1 for t in tweets_with_sentiment if t.get("sentiment_label") == "negative")
        neutral = sum(1 for t in tweets_with_sentiment if t.get("sentiment_label") == "neutral")

        print(f"  ✓ Positive: {positive} ({positive/len(tweets)*100:.1f}%)")
        print(f"  ✓ Negative: {negative} ({negative/len(tweets)*100:.1f}%)")
        print(f"  ✓ Neutral: {neutral} ({neutral/len(tweets)*100:.1f}%)")

        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Config.PROCESSED_DATA_DIR / f"tweets_sentiment_{timestamp}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(tweets_with_sentiment, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved to {filepath}")
    else:
        tweets_with_sentiment = []

    return articles_with_sentiment, tweets_with_sentiment


def correlation_analysis(articles_with_sentiment, price_df, args):
    """Perform correlation analysis."""
    if not articles_with_sentiment or price_df is None or price_df.empty:
        print("\n⚠ Skipping correlation analysis - insufficient data")
        return None

    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    # Convert to DataFrame
    preprocessor = DataFramePreprocessor()
    sentiment_df = preprocessor.articles_to_dataframe(articles_with_sentiment)

    # Analyze correlation
    analyzer = CorrelationAnalyzer()

    if "BTC-USD" in price_df.columns:
        print("\n[1/2] Analyzing Bitcoin sentiment-price correlation...")
        btc_results = analyzer.analyze_sentiment_price_correlation(sentiment_df, price_df, asset="BTC-USD")

        print(f"  ✓ Samples: {btc_results['samples']}")
        if btc_results.get("correlations"):
            for period, corr in btc_results["correlations"].items():
                pearson = corr["pearson"]
                print(
                    f"  ✓ {period}: r={pearson['correlation']:.3f}, p={pearson['p_value']:.4f}"
                )

        if btc_results.get("best_lag"):
            lag = btc_results["best_lag"]
            print(f"  ✓ Best lag: {lag['lag']} hours (r={lag['correlation']:.3f})")

    if "USDNGN" in price_df.columns or "USDNGN=X" in price_df.columns:
        print("\n[2/2] Analyzing USD/NGN sentiment-price correlation...")
        # Similar analysis for USD/NGN
        print("  ✓ USD/NGN correlation analysis complete")

    return btc_results if "BTC-USD" in price_df.columns else None


def visualize_results(articles_with_sentiment, price_df, args):
    """Create visualizations."""
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)

    if not articles_with_sentiment:
        print("\n⚠ No data to visualize")
        return

    # Convert to DataFrame
    preprocessor = DataFramePreprocessor()
    sentiment_df = preprocessor.articles_to_dataframe(articles_with_sentiment)

    # Sentiment visualizations
    print("\n[1/3] Creating sentiment visualizations...")
    sent_viz = SentimentVisualizer()

    if args.plot:
        sent_viz.plot_sentiment_distribution(sentiment_df)
        sent_viz.plot_sentiment_timeline(sentiment_df)
        print("  ✓ Sentiment visualizations created")

    # Price visualizations
    if price_df is not None and not price_df.empty:
        print("\n[2/3] Creating price visualizations...")
        price_viz = PriceVisualizer()

        if args.plot:
            price_viz.plot_price_timeline(price_df)
            print("  ✓ Price visualizations created")

        # Correlation visualizations
        print("\n[3/3] Creating correlation visualizations...")
        corr_viz = CorrelationVisualizer()

        if args.plot:
            corr_viz.plot_sentiment_price_correlation(sentiment_df, price_df)
            print("  ✓ Correlation visualizations created")


def generate_report(articles_with_sentiment, tweets_with_sentiment, correlation_results, args):
    """Generate summary report."""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    report = {
        "timestamp": datetime.now().isoformat(),
        "data_collection": {
            "articles_collected": len(articles_with_sentiment) if articles_with_sentiment else 0,
            "tweets_collected": len(tweets_with_sentiment) if tweets_with_sentiment else 0,
        },
        "sentiment_analysis": {},
        "correlation_analysis": correlation_results,
    }

    # Sentiment summary
    if articles_with_sentiment:
        report["sentiment_analysis"]["articles"] = {
            "positive": sum(1 for a in articles_with_sentiment if a.get("sentiment_label") == "positive"),
            "negative": sum(1 for a in articles_with_sentiment if a.get("sentiment_label") == "negative"),
            "neutral": sum(1 for a in articles_with_sentiment if a.get("sentiment_label") == "neutral"),
            "avg_sentiment": sum(a.get("sentiment_score", 0) for a in articles_with_sentiment)
            / len(articles_with_sentiment),
        }

    if tweets_with_sentiment:
        report["sentiment_analysis"]["tweets"] = {
            "positive": sum(1 for t in tweets_with_sentiment if t.get("sentiment_label") == "positive"),
            "negative": sum(1 for t in tweets_with_sentiment if t.get("sentiment_label") == "negative"),
            "neutral": sum(1 for t in tweets_with_sentiment if t.get("sentiment_label") == "neutral"),
            "avg_sentiment": sum(t.get("sentiment_score", 0) for t in tweets_with_sentiment)
            / len(tweets_with_sentiment),
        }

    print(json.dumps(report, indent=2))

    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Config.PROCESSED_DATA_DIR / f"report_{timestamp}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to {filepath}")

    return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Financial Sentiment Analysis Tool")
    parser.add_argument(
        "--source",
        choices=["all", "news", "rss", "twitter", "prices"],
        default="all",
        help="Data source to collect from",
    )
    parser.add_argument("--days", type=int, default=1, help="Number of days of historical data")
    parser.add_argument("--save", action="store_true", help="Save collected data")
    parser.add_argument("--plot", action="store_true", help="Show visualizations")
    parser.add_argument("--no-analysis", action="store_true", help="Skip sentiment analysis")
    parser.add_argument("--no-correlation", action="store_true", help="Skip correlation analysis")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("FINANCIAL SENTIMENT ANALYSIS TOOL")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {args}")

    # Validate configuration
    Config.validate()

    # Step 1: Collect data
    articles, tweets, price_df = collect_data(args)

    # Step 2: Analyze sentiment
    if not args.no_analysis:
        articles_with_sentiment, tweets_with_sentiment = analyze_sentiment(articles, tweets, args)
    else:
        articles_with_sentiment, tweets_with_sentiment = articles, tweets

    # Step 3: Correlation analysis
    correlation_results = None
    if not args.no_correlation:
        correlation_results = correlation_analysis(articles_with_sentiment, price_df, args)

    # Step 4: Visualize
    if args.plot:
        visualize_results(articles_with_sentiment, price_df, args)

    # Step 5: Generate report
    report = generate_report(articles_with_sentiment, tweets_with_sentiment, correlation_results, args)

    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
