#!/bin/bash
# Setup script for Financial Sentiment Analysis Tool

echo "========================================"
echo "Financial Sentiment Analysis Setup"
echo "========================================"

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys!"
fi

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/raw data/processed data/external models

echo ""
echo "========================================"
echo "✓ Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit .env file with your API keys"
echo "3. Run the demo: python scripts/run_analysis.py --help"
echo "4. Or open the Jupyter notebook: jupyter notebook notebooks/01_demo_sentiment_analysis.ipynb"
echo ""
