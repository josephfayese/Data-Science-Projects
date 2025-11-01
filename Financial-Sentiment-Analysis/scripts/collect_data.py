#!/usr/bin/env python
"""
Simple script to collect data from all sources.
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import Config
from src.data.collectors import NewsCollector, RSSFeedCollector, PriceCollector
from src.data.twitter_collector import TwitterCollector


def main():
    parser = argparse.ArgumentParser(description="Collect financial data")
    parser.add_argument("--days", type=int, default=1, help="Days of historical data")
    parser.add_argument("--keywords", nargs="+", default=None, help="Keywords to search")
    args = parser.parse_args()

    keywords = args.keywords or Config.ALL_KEYWORDS[:3]

    print("=" * 80)
    print("DATA COLLECTION TOOL")
    print("=" * 80)

    # Collect news
    print("\n[1/4] Collecting news articles...")
    news_collector = NewsCollector()
    news = news_collector.collect_news(
        keywords=keywords, from_date=datetime.now() - timedelta(days=args.days)
    )
    print(f"  ✓ Collected {len(news)} articles")
    if news:
        news_collector.save_to_json(news)

    # Collect RSS
    print("\n[2/4] Collecting RSS feeds...")
    rss_collector = RSSFeedCollector()
    rss = rss_collector.collect_all_feeds(max_entries_per_feed=20)
    print(f"  ✓ Collected {len(rss)} articles")
    if rss:
        rss_collector.save_to_json(rss)

    # Collect Twitter
    print("\n[3/4] Collecting Twitter data...")
    twitter_collector = TwitterCollector()
    if twitter_collector.client:
        tweets = twitter_collector.collect_all(max_results_per_topic=100)
        print(f"  ✓ Collected {len(tweets)} tweets")
        if tweets:
            twitter_collector.save_to_json(tweets)
    else:
        print("  ✗ Twitter API not configured")

    # Collect prices
    print("\n[4/4] Collecting price data...")
    price_collector = PriceCollector()
    prices = price_collector.get_historical_prices(period=f"{args.days}d")
    print(f"  ✓ Collected {len(prices)} price records")
    if not prices.empty:
        price_collector.save_to_csv(prices)

    print("\n" + "=" * 80)
    print("✓ Data collection complete!")
    print(f"Data saved to: {Config.RAW_DATA_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
