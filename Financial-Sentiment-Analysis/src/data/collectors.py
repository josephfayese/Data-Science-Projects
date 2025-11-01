"""
Data collectors for news, social media, and financial data.
"""

import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from newsapi import NewsApiClient
import feedparser
import yfinance as yf
import pandas as pd
from pathlib import Path

from ..config.settings import Config


class NewsCollector:
    """Collects news articles from NewsAPI and other sources."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsCollector.

        Args:
            api_key: NewsAPI key. If None, uses key from Config.
        """
        self.api_key = api_key or Config.NEWS_API_KEY
        if self.api_key:
            self.client = NewsApiClient(api_key=self.api_key)
        else:
            self.client = None
            print("Warning: NewsAPI key not provided. NewsAPI features will be disabled.")

    def collect_news(
        self,
        keywords: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
    ) -> List[Dict]:
        """
        Collect news articles from NewsAPI.

        Args:
            keywords: List of keywords to search for.
            sources: List of news sources.
            from_date: Start date for articles.
            to_date: End date for articles.
            language: Language code (default: 'en').

        Returns:
            List of article dictionaries.
        """
        if not self.client:
            print("NewsAPI client not initialized. Skipping collection.")
            return []

        keywords = keywords or Config.ALL_KEYWORDS
        sources = sources or Config.NEWS_SOURCES

        articles = []

        # Set default date range (last 24 hours)
        if not from_date:
            from_date = datetime.now() - timedelta(days=1)
        if not to_date:
            to_date = datetime.now()

        # Search for each keyword
        for keyword in keywords[:5]:  # Limit to avoid rate limits
            try:
                query = f"{keyword} AND (policy OR announcement OR regulation OR government)"

                response = self.client.get_everything(
                    q=query,
                    sources=",".join(sources) if sources else None,
                    from_param=from_date.strftime("%Y-%m-%d"),
                    to=to_date.strftime("%Y-%m-%d"),
                    language=language,
                    sort_by="publishedAt",
                    page_size=Config.MAX_ARTICLES_PER_REQUEST // len(keywords[:5]),
                )

                if response.get("status") == "ok":
                    for article in response.get("articles", []):
                        articles.append(
                            {
                                "source": article.get("source", {}).get("name", "Unknown"),
                                "author": article.get("author"),
                                "title": article.get("title"),
                                "description": article.get("description"),
                                "url": article.get("url"),
                                "published_at": article.get("publishedAt"),
                                "content": article.get("content"),
                                "keyword": keyword,
                                "collected_at": datetime.now().isoformat(),
                            }
                        )

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"Error collecting news for keyword '{keyword}': {str(e)}")

        return articles

    def save_to_json(self, articles: List[Dict], filename: Optional[str] = None):
        """
        Save articles to JSON file.

        Args:
            articles: List of article dictionaries.
            filename: Output filename. If None, uses timestamp.
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_{timestamp}.json"

        filepath = Config.RAW_DATA_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(articles)} articles to {filepath}")


class RSSFeedCollector:
    """Collects articles from RSS feeds."""

    def __init__(self, feed_urls: Optional[List[str]] = None):
        """
        Initialize RSSFeedCollector.

        Args:
            feed_urls: List of RSS feed URLs.
        """
        self.feed_urls = feed_urls or (Config.RSS_FEEDS + Config.NIGERIAN_NEWS_SOURCES)

    def collect_from_feed(self, feed_url: str, max_entries: int = 50) -> List[Dict]:
        """
        Collect articles from a single RSS feed.

        Args:
            feed_url: RSS feed URL.
            max_entries: Maximum number of entries to collect.

        Returns:
            List of article dictionaries.
        """
        articles = []

        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:max_entries]:
                articles.append(
                    {
                        "source": feed.feed.get("title", feed_url),
                        "title": entry.get("title"),
                        "description": entry.get("summary", entry.get("description")),
                        "url": entry.get("link"),
                        "published_at": entry.get("published", entry.get("updated")),
                        "content": entry.get("content", [{}])[0].get("value")
                        if entry.get("content")
                        else None,
                        "feed_url": feed_url,
                        "collected_at": datetime.now().isoformat(),
                    }
                )

        except Exception as e:
            print(f"Error collecting from feed {feed_url}: {str(e)}")

        return articles

    def collect_all_feeds(self, max_entries_per_feed: int = 50) -> List[Dict]:
        """
        Collect articles from all configured RSS feeds.

        Args:
            max_entries_per_feed: Maximum entries per feed.

        Returns:
            List of all articles.
        """
        all_articles = []

        for feed_url in self.feed_urls:
            print(f"Collecting from {feed_url}...")
            articles = self.collect_from_feed(feed_url, max_entries_per_feed)
            all_articles.extend(articles)
            time.sleep(1)  # Rate limiting

        return all_articles

    def save_to_json(self, articles: List[Dict], filename: Optional[str] = None):
        """Save articles to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rss_{timestamp}.json"

        filepath = Config.RAW_DATA_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(articles)} articles to {filepath}")


class PriceCollector:
    """Collects cryptocurrency and FX price data."""

    def __init__(self):
        """Initialize PriceCollector."""
        self.bitcoin_symbols = Config.BITCOIN_SYMBOLS
        self.fx_symbols = Config.FX_SYMBOLS

    def get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for Bitcoin and USD/NGN.

        Returns:
            Dictionary of symbol: price pairs.
        """
        prices = {}

        # Get Bitcoin price
        for symbol in self.bitcoin_symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                if not data.empty:
                    prices[symbol] = float(data["Close"].iloc[-1])
                    break
            except Exception as e:
                print(f"Error fetching {symbol}: {str(e)}")

        # Get USD/NGN rate
        for symbol in self.fx_symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                if not data.empty:
                    prices[symbol] = float(data["Close"].iloc[-1])
                else:
                    # Fallback: Use a fixed rate or API
                    print(f"No data for {symbol}, using fallback...")
                    prices["USDNGN"] = self._get_usdngn_fallback()
                break
            except Exception as e:
                print(f"Error fetching {symbol}: {str(e)}")
                prices["USDNGN"] = self._get_usdngn_fallback()

        prices["timestamp"] = datetime.now().isoformat()

        return prices

    def _get_usdngn_fallback(self) -> float:
        """
        Get USD/NGN rate from alternative source.

        Returns:
            USD/NGN exchange rate.
        """
        try:
            # Try Alpha Vantage
            if Config.ALPHA_VANTAGE_KEY:
                url = f"https://www.alphavantage.co/query"
                params = {
                    "function": "CURRENCY_EXCHANGE_RATE",
                    "from_currency": "USD",
                    "to_currency": "NGN",
                    "apikey": Config.ALPHA_VANTAGE_KEY,
                }
                response = requests.get(url, params=params)
                data = response.json()
                rate = float(
                    data.get("Realtime Currency Exchange Rate", {}).get(
                        "5. Exchange Rate", 0
                    )
                )
                if rate > 0:
                    return rate
        except Exception as e:
            print(f"Error getting USD/NGN from Alpha Vantage: {str(e)}")

        # Default fallback rate (update manually or use another API)
        return 1630.0  # Approximate rate as of late 2024

    def get_historical_prices(
        self, symbols: Optional[List[str]] = None, period: str = "1mo"
    ) -> pd.DataFrame:
        """
        Get historical price data.

        Args:
            symbols: List of symbols to fetch.
            period: Time period (e.g., '1d', '1mo', '1y').

        Returns:
            DataFrame with historical prices.
        """
        if symbols is None:
            symbols = self.bitcoin_symbols + self.fx_symbols

        all_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                if not data.empty:
                    all_data[symbol] = data["Close"]
            except Exception as e:
                print(f"Error fetching historical data for {symbol}: {str(e)}")

        if all_data:
            df = pd.DataFrame(all_data)
            return df
        else:
            return pd.DataFrame()

    def save_to_csv(self, data: pd.DataFrame, filename: Optional[str] = None):
        """Save price data to CSV file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prices_{timestamp}.csv"

        filepath = Config.RAW_DATA_DIR / filename
        data.to_csv(filepath)

        print(f"Saved price data to {filepath}")


# Example usage
if __name__ == "__main__":
    # Test NewsCollector
    print("Testing NewsCollector...")
    news_collector = NewsCollector()
    articles = news_collector.collect_news(keywords=["Bitcoin", "Nigeria"])
    print(f"Collected {len(articles)} news articles")
    if articles:
        news_collector.save_to_json(articles)

    # Test RSSFeedCollector
    print("\nTesting RSSFeedCollector...")
    rss_collector = RSSFeedCollector()
    rss_articles = rss_collector.collect_all_feeds(max_entries_per_feed=10)
    print(f"Collected {len(rss_articles)} RSS articles")
    if rss_articles:
        rss_collector.save_to_json(rss_articles)

    # Test PriceCollector
    print("\nTesting PriceCollector...")
    price_collector = PriceCollector()
    current_prices = price_collector.get_current_prices()
    print(f"Current prices: {current_prices}")

    historical_prices = price_collector.get_historical_prices(period="7d")
    print(f"Historical prices shape: {historical_prices.shape}")
    if not historical_prices.empty:
        price_collector.save_to_csv(historical_prices)
