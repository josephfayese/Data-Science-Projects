# Quick Start Guide

Get started with the Financial Sentiment Analysis Tool in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

## Installation

### Option 1: Automatic Setup (Recommended)

```bash
cd Financial-Sentiment-Analysis
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"

# Create .env file
cp .env.example .env
```

## Configuration

Edit the `.env` file with your API keys:

```bash
# Required for news collection
NEWS_API_KEY=your_key_here

# Optional for Twitter collection
TWITTER_BEARER_TOKEN=your_token_here

# Optional for enhanced FX data
ALPHA_VANTAGE_KEY=your_key_here
```

### Getting API Keys

1. **NewsAPI** (Free): https://newsapi.org/register
   - Free tier: 100 requests/day

2. **Twitter API** (Optional): https://developer.twitter.com/
   - Apply for developer account
   - Get Bearer Token from your app

3. **Alpha Vantage** (Optional): https://www.alphavantage.co/support/#api-key
   - Free tier: 25 requests/day

## Usage

### 1. Run Complete Analysis

```bash
# Activate virtual environment
source venv/bin/activate

# Run analysis with all sources
python scripts/run_analysis.py --source all --days 1 --save --plot

# Run analysis with specific source
python scripts/run_analysis.py --source news --days 3 --save
```

### 2. Collect Data Only

```bash
python scripts/collect_data.py --days 7 --keywords Bitcoin Nigeria
```

### 3. Interactive Analysis (Jupyter Notebook)

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/01_demo_sentiment_analysis.ipynb
```

## Example Output

```
================================================================================
DATA COLLECTION
================================================================================

[1/4] Collecting news articles...
  ✓ Collected 45 news articles

[2/4] Collecting from RSS feeds...
  ✓ Collected 78 RSS articles

[3/4] Collecting Twitter data...
  ✓ Collected 134 tweets

[4/4] Collecting price data...
  ✓ Current prices: {'BTC-USD': 67892.50, 'USDNGN': 1630.00}

================================================================================
SENTIMENT ANALYSIS
================================================================================

[1/2] Analyzing 123 articles...
  ✓ Positive: 45 (36.6%)
  ✓ Negative: 32 (26.0%)
  ✓ Neutral: 46 (37.4%)

================================================================================
CORRELATION ANALYSIS
================================================================================

[1/2] Analyzing Bitcoin sentiment-price correlation...
  ✓ Samples: 98
  ✓ 1h: r=0.234, p=0.0234
  ✓ 3h: r=0.312, p=0.0045
  ✓ Best lag: 2 hours (r=0.367)
```

## Command Line Options

```bash
python scripts/run_analysis.py --help

Options:
  --source {all,news,rss,twitter,prices}  Data source to collect from
  --days INT                               Number of days of historical data
  --save                                   Save collected data
  --plot                                   Show visualizations
  --no-analysis                           Skip sentiment analysis
  --no-correlation                        Skip correlation analysis
```

## Sample Code

```python
from src.data.collectors import NewsCollector, PriceCollector
from src.sentiment.analyzers import EnsembleAnalyzer

# Collect news
collector = NewsCollector()
articles = collector.collect_news(keywords=["Bitcoin"])

# Analyze sentiment
analyzer = EnsembleAnalyzer()
for article in articles:
    text = article.get('title', '') + ' ' + article.get('description', '')
    sentiment = analyzer.analyze(text)
    print(f"Score: {sentiment['ensemble_score']:.3f}, Label: {sentiment['ensemble_label']}")

# Get prices
price_collector = PriceCollector()
prices = price_collector.get_current_prices()
print(f"Bitcoin: ${prices.get('BTC-USD', 0):,.2f}")
```

## Troubleshooting

### Issue: "No module named 'src'"

**Solution:** Make sure you're running scripts from the project root:
```bash
cd Financial-Sentiment-Analysis
python scripts/run_analysis.py
```

### Issue: "API key not found"

**Solution:** Ensure your `.env` file exists and contains valid API keys:
```bash
cat .env
```

### Issue: "No articles collected"

**Solution:**
- Check your API keys are valid
- Verify you have internet connection
- Try with fewer keywords
- Check API rate limits

### Issue: "Twitter API not available"

**Solution:** Twitter API is optional. The tool works without it. To enable:
1. Get Twitter Developer account
2. Add `TWITTER_BEARER_TOKEN` to `.env`

## What's Next?

1. **Customize Keywords**: Edit `src/config/settings.py` to add your own keywords
2. **Add Data Sources**: Add new RSS feeds in `src/config/settings.py`
3. **Extend Analysis**: Modify `src/sentiment/analyzers.py` for custom sentiment logic
4. **Schedule Collection**: Use cron or Task Scheduler to run collection periodically
5. **Build Dashboard**: Create a web dashboard using the API module

## Getting Help

- Read the full [README.md](README.md)
- Check the [Problem Statement](Problem%20Statement.txt)
- Review example [Jupyter notebook](notebooks/01_demo_sentiment_analysis.ipynb)
- Open an issue on GitHub

## Resources

- [NewsAPI Documentation](https://newsapi.org/docs)
- [Twitter API Documentation](https://developer.twitter.com/en/docs)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
