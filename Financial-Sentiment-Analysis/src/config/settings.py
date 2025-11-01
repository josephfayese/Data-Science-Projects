"""
Configuration settings for the Financial Sentiment Analysis tool.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model directory
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/sentiment.db")

# Data Collection Settings
DATA_COLLECTION_INTERVAL = int(os.getenv("DATA_COLLECTION_INTERVAL", 300))  # 5 minutes
MAX_ARTICLES_PER_REQUEST = int(os.getenv("MAX_ARTICLES_PER_REQUEST", 100))

# Keywords for monitoring
BITCOIN_KEYWORDS = os.getenv("BITCOIN_KEYWORDS", "Bitcoin,BTC,cryptocurrency,crypto").split(",")
NIGERIA_KEYWORDS = os.getenv("NIGERIA_KEYWORDS", "Nigeria,Naira,NGN,CBN,Central Bank of Nigeria").split(",")
FX_KEYWORDS = os.getenv("FX_KEYWORDS", "forex,exchange rate,USD/NGN,USDNGN").split(",")

# Combined keywords
ALL_KEYWORDS = list(set(BITCOIN_KEYWORDS + NIGERIA_KEYWORDS + FX_KEYWORDS))

# Timezone
TIMEZONE = os.getenv("TIMEZONE", "Africa/Lagos")

# News Sources
NEWS_SOURCES = [
    "bloomberg",
    "financial-times",
    "the-wall-street-journal",
    "reuters",
    "cnbc",
    "business-insider",
]

# RSS Feed URLs
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://www.reuters.com/business/finance",
]

# Nigerian News Sources
NIGERIAN_NEWS_SOURCES = [
    "https://punchng.com/feed/",
    "https://guardian.ng/feed/",
    "https://www.vanguardngr.com/feed/",
    "https://businessday.ng/feed/",
]

# Price Data Sources
BITCOIN_SYMBOLS = ["BTC-USD", "BTCUSDT"]
FX_SYMBOLS = ["USDNGN=X"]

# Sentiment Analysis Settings
SENTIMENT_MODELS = ["vader", "textblob", "finbert"]
DEFAULT_SENTIMENT_MODEL = "vader"

# Correlation Analysis Settings
CORRELATION_WINDOW = 24  # hours
MIN_CORRELATION_THRESHOLD = 0.3

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Alert Settings
ALERT_THRESHOLD_SENTIMENT = 0.7  # Trigger alert for strong sentiment
ALERT_THRESHOLD_PRICE_CHANGE = 0.05  # 5% price change
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")

# Rate Limiting
API_RATE_LIMIT_DELAY = 1  # seconds between API calls
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


class Config:
    """Configuration class for easy access to settings."""

    # Directories
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    RAW_DATA_DIR = RAW_DATA_DIR
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
    EXTERNAL_DATA_DIR = EXTERNAL_DATA_DIR
    MODELS_DIR = MODELS_DIR

    # API Keys
    NEWS_API_KEY = NEWS_API_KEY
    TWITTER_API_KEY = TWITTER_API_KEY
    TWITTER_API_SECRET = TWITTER_API_SECRET
    TWITTER_ACCESS_TOKEN = TWITTER_ACCESS_TOKEN
    TWITTER_ACCESS_SECRET = TWITTER_ACCESS_SECRET
    TWITTER_BEARER_TOKEN = TWITTER_BEARER_TOKEN
    ALPHA_VANTAGE_KEY = ALPHA_VANTAGE_KEY

    # Database
    DATABASE_URL = DATABASE_URL

    # Keywords
    BITCOIN_KEYWORDS = BITCOIN_KEYWORDS
    NIGERIA_KEYWORDS = NIGERIA_KEYWORDS
    FX_KEYWORDS = FX_KEYWORDS
    ALL_KEYWORDS = ALL_KEYWORDS

    # Sources
    NEWS_SOURCES = NEWS_SOURCES
    RSS_FEEDS = RSS_FEEDS
    NIGERIAN_NEWS_SOURCES = NIGERIAN_NEWS_SOURCES

    # Symbols
    BITCOIN_SYMBOLS = BITCOIN_SYMBOLS
    FX_SYMBOLS = FX_SYMBOLS

    # Settings
    DATA_COLLECTION_INTERVAL = DATA_COLLECTION_INTERVAL
    MAX_ARTICLES_PER_REQUEST = MAX_ARTICLES_PER_REQUEST
    TIMEZONE = TIMEZONE
    LOG_LEVEL = LOG_LEVEL

    @classmethod
    def validate(cls) -> bool:
        """Validate that required API keys are set."""
        required_keys = {
            "NEWS_API_KEY": cls.NEWS_API_KEY,
        }

        missing_keys = [key for key, value in required_keys.items() if not value]

        if missing_keys:
            print(f"Warning: Missing API keys: {', '.join(missing_keys)}")
            print("Some features may not work without proper API keys.")
            return False

        return True


# Validate configuration on import
if __name__ == "__main__":
    Config.validate()
    print("Configuration loaded successfully!")
    print(f"Base directory: {Config.BASE_DIR}")
    print(f"Keywords: {Config.ALL_KEYWORDS}")
