# Financial Sentiment Analysis Tool

A comprehensive tool for analyzing news, social media, and blog sentiment to predict potential impacts on Bitcoin and USD/NGN exchange rates.

## Overview

This tool automatically collects and analyzes content from multiple sources including news feeds, Twitter/X, and financial blogs to identify policies and announcements that could impact Bitcoin prices and the USD/NGN exchange rate in Nigeria.

## Features

- **Multi-Source Data Collection**: Gather data from news APIs, Twitter, RSS feeds, and blogs
- **Sentiment Analysis**: Advanced NLP using VADER, TextBlob, and FinBERT transformer models
- **Price Tracking**: Real-time Bitcoin and USD/NGN exchange rate monitoring
- **Correlation Analysis**: Statistical analysis of sentiment-price relationships
- **Visualization**: Interactive dashboards and reports
- **Alert System**: Automated notifications for significant market events

## Project Structure

```
Financial-Sentiment-Analysis/
├── src/                      # Source code
│   ├── config/              # Configuration and settings
│   ├── data/                # Data collection modules
│   ├── sentiment/           # Sentiment analysis models
│   ├── features/            # Feature engineering
│   ├── models/              # ML models
│   ├── utils/               # Utility functions
│   └── api/                 # API endpoints (optional)
├── data/                    # Data storage
│   ├── raw/                 # Raw collected data
│   ├── processed/           # Processed datasets
│   └── external/            # External reference data
├── notebooks/               # Jupyter notebooks for analysis
├── models/                  # Trained models
├── tests/                   # Unit tests
├── scripts/                 # Standalone scripts
└── config/                  # Configuration files
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Financial-Sentiment-Analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with the following API keys:

```env
# News API
NEWS_API_KEY=your_newsapi_key

# Twitter API
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_BEARER_TOKEN=your_bearer_token

# Alpha Vantage (for FX data)
ALPHA_VANTAGE_KEY=your_alpha_vantage_key

# Database (optional)
DATABASE_URL=sqlite:///data/sentiment.db
```

## Usage

### 1. Collect Data

```bash
python scripts/collect_data.py --source all --duration 24h
```

### 2. Run Sentiment Analysis

```bash
python scripts/run_analysis.py --input data/raw/news.json --output data/processed/sentiment.csv
```

### 3. Generate Reports

```bash
python scripts/generate_report.py --date 2025-11-01
```

### 4. Interactive Analysis

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

## Data Sources

### News Sources
- NewsAPI (newsapi.org)
- Financial Times RSS feeds
- Bloomberg RSS feeds
- Local Nigerian news sources

### Social Media
- Twitter/X (keywords: Bitcoin, BTC, Naira, USDNGN, CBN)

### Financial Data
- Bitcoin prices: CoinGecko, Binance, yfinance
- USD/NGN rates: Alpha Vantage, Central Bank of Nigeria

## Sentiment Analysis Models

1. **VADER**: Rule-based sentiment analysis optimized for social media
2. **TextBlob**: Simple pattern-based sentiment analysis
3. **FinBERT**: Pre-trained transformer model for financial text

## Key Modules

### Data Collection (`src/data/collectors.py`)
- `NewsCollector`: Fetches news from various APIs
- `TwitterCollector`: Collects tweets using Twitter API
- `BlogCollector`: Scrapes RSS feeds
- `PriceCollector`: Fetches real-time price data

### Sentiment Analysis (`src/sentiment/analyzers.py`)
- `VaderAnalyzer`: VADER sentiment analysis
- `TransformerAnalyzer`: FinBERT-based analysis
- `EnsembleAnalyzer`: Combines multiple models

### Correlation Analysis (`src/models/correlation.py`)
- Statistical correlation between sentiment and prices
- Time-lagged correlation analysis
- Event impact analysis

## Example Workflow

```python
from src.data.collectors import NewsCollector, PriceCollector
from src.sentiment.analyzers import EnsembleAnalyzer
from src.models.correlation import CorrelationAnalyzer

# Collect data
news_collector = NewsCollector()
price_collector = PriceCollector()

news_data = news_collector.collect(keywords=['Bitcoin', 'Nigeria', 'CBN'])
price_data = price_collector.get_prices(['BTC-USD', 'USD-NGN'])

# Analyze sentiment
analyzer = EnsembleAnalyzer()
sentiment_scores = analyzer.analyze(news_data)

# Correlate with prices
correlator = CorrelationAnalyzer()
correlation = correlator.analyze(sentiment_scores, price_data)

print(f"Sentiment-Price Correlation: {correlation.coefficient}")
```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=src tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License

## Contact

For questions or support, please open an issue on GitHub.

## Acknowledgments

- NewsAPI for news data
- Twitter API for social media data
- HuggingFace for transformer models
- FinBERT model for financial sentiment analysis
