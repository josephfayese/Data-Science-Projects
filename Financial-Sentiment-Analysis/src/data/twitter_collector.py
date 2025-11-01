"""
Twitter/X data collector for social media sentiment analysis.
"""

import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import tweepy

from ..config.settings import Config


class TwitterCollector:
    """Collects tweets related to Bitcoin and Nigeria FX."""

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_secret: Optional[str] = None,
    ):
        """
        Initialize TwitterCollector.

        Args:
            bearer_token: Twitter API Bearer Token.
            api_key: Twitter API Key.
            api_secret: Twitter API Secret.
            access_token: Twitter Access Token.
            access_secret: Twitter Access Token Secret.
        """
        self.bearer_token = bearer_token or Config.TWITTER_BEARER_TOKEN
        self.api_key = api_key or Config.TWITTER_API_KEY
        self.api_secret = api_secret or Config.TWITTER_API_SECRET
        self.access_token = access_token or Config.TWITTER_ACCESS_TOKEN
        self.access_secret = access_secret or Config.TWITTER_ACCESS_SECRET

        self.client = None

        # Initialize client based on available credentials
        if self.bearer_token:
            try:
                self.client = tweepy.Client(bearer_token=self.bearer_token)
                print("Twitter API v2 client initialized successfully")
            except Exception as e:
                print(f"Error initializing Twitter client: {str(e)}")
        else:
            print("Warning: Twitter credentials not provided. Twitter features will be disabled.")

    def search_tweets(
        self,
        query: str,
        max_results: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Search for tweets matching a query.

        Args:
            query: Search query string.
            max_results: Maximum number of tweets to return (10-100).
            start_time: Start time for search.
            end_time: End time for search.

        Returns:
            List of tweet dictionaries.
        """
        if not self.client:
            print("Twitter client not initialized. Skipping collection.")
            return []

        tweets = []

        try:
            # Set default time range (last 7 days for standard access)
            if not start_time:
                start_time = datetime.now() - timedelta(days=7)
            if not end_time:
                end_time = datetime.now()

            # Search recent tweets
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                start_time=start_time,
                end_time=end_time,
                tweet_fields=["created_at", "author_id", "public_metrics", "lang", "source"],
                expansions=["author_id"],
                user_fields=["username", "verified", "public_metrics"],
            )

            if response.data:
                # Create user lookup dictionary
                users = {}
                if response.includes and "users" in response.includes:
                    users = {user.id: user for user in response.includes["users"]}

                for tweet in response.data:
                    author = users.get(tweet.author_id)

                    tweets.append(
                        {
                            "id": tweet.id,
                            "text": tweet.text,
                            "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                            "author_id": tweet.author_id,
                            "author_username": author.username if author else None,
                            "author_verified": author.verified if author else False,
                            "author_followers": (
                                author.public_metrics["followers_count"] if author else 0
                            ),
                            "retweet_count": tweet.public_metrics.get("retweet_count", 0),
                            "reply_count": tweet.public_metrics.get("reply_count", 0),
                            "like_count": tweet.public_metrics.get("like_count", 0),
                            "quote_count": tweet.public_metrics.get("quote_count", 0),
                            "lang": tweet.lang,
                            "source": tweet.source if hasattr(tweet, "source") else None,
                            "query": query,
                            "collected_at": datetime.now().isoformat(),
                        }
                    )

            print(f"Collected {len(tweets)} tweets for query: {query}")

        except tweepy.errors.TweepyException as e:
            print(f"Twitter API error for query '{query}': {str(e)}")
        except Exception as e:
            print(f"Error collecting tweets for query '{query}': {str(e)}")

        return tweets

    def collect_bitcoin_tweets(self, max_results: int = 100) -> List[Dict]:
        """
        Collect tweets about Bitcoin.

        Args:
            max_results: Maximum number of tweets per query.

        Returns:
            List of tweet dictionaries.
        """
        queries = [
            "Bitcoin (policy OR regulation OR announcement OR government) -is:retweet lang:en",
            "BTC (CBN OR Nigeria OR Naira) -is:retweet lang:en",
            "cryptocurrency (regulation OR ban OR policy) Nigeria -is:retweet lang:en",
        ]

        all_tweets = []

        for query in queries:
            tweets = self.search_tweets(query, max_results=max_results // len(queries))
            all_tweets.extend(tweets)
            time.sleep(2)  # Rate limiting

        return all_tweets

    def collect_nigeria_fx_tweets(self, max_results: int = 100) -> List[Dict]:
        """
        Collect tweets about Nigeria FX and USD/NGN.

        Args:
            max_results: Maximum number of tweets per query.

        Returns:
            List of tweet dictionaries.
        """
        queries = [
            "(USD NGN OR USDNGN OR Naira) (CBN OR policy OR rate) -is:retweet lang:en",
            "(Nigeria OR Naira) (forex OR FX OR exchange rate) -is:retweet lang:en",
            "CBN (policy OR announcement OR rate) -is:retweet lang:en",
        ]

        all_tweets = []

        for query in queries:
            tweets = self.search_tweets(query, max_results=max_results // len(queries))
            all_tweets.extend(tweets)
            time.sleep(2)  # Rate limiting

        return all_tweets

    def collect_all(self, max_results_per_topic: int = 100) -> List[Dict]:
        """
        Collect tweets for all topics (Bitcoin and Nigeria FX).

        Args:
            max_results_per_topic: Maximum results per topic.

        Returns:
            List of all collected tweets.
        """
        all_tweets = []

        print("Collecting Bitcoin-related tweets...")
        bitcoin_tweets = self.collect_bitcoin_tweets(max_results_per_topic)
        all_tweets.extend(bitcoin_tweets)

        print("Collecting Nigeria FX-related tweets...")
        fx_tweets = self.collect_nigeria_fx_tweets(max_results_per_topic)
        all_tweets.extend(fx_tweets)

        # Remove duplicates based on tweet ID
        unique_tweets = {tweet["id"]: tweet for tweet in all_tweets}.values()
        unique_tweets = list(unique_tweets)

        print(f"Total unique tweets collected: {len(unique_tweets)}")

        return unique_tweets

    def save_to_json(self, tweets: List[Dict], filename: Optional[str] = None):
        """
        Save tweets to JSON file.

        Args:
            tweets: List of tweet dictionaries.
            filename: Output filename. If None, uses timestamp.
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tweets_{timestamp}.json"

        filepath = Config.RAW_DATA_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(tweets, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(tweets)} tweets to {filepath}")


# Example usage
if __name__ == "__main__":
    # Test TwitterCollector
    print("Testing TwitterCollector...")
    twitter_collector = TwitterCollector()

    if twitter_collector.client:
        # Collect all tweets
        tweets = twitter_collector.collect_all(max_results_per_topic=50)

        if tweets:
            print(f"\nSample tweet:")
            print(json.dumps(tweets[0], indent=2))

            # Save to file
            twitter_collector.save_to_json(tweets)
    else:
        print("Twitter client not available. Please configure API credentials.")
