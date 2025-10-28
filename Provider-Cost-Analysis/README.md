# Provider Cost Analysis Across Currency Corridors

This project provides SQL queries and Python code to capture and analyze provider costs across all currency corridors.

## Overview

The solution includes:
- **SQL Queries**: Comprehensive queries to extract provider costs data
- **Python Analysis Tool**: A complete class-based system for querying, analyzing, and exporting provider cost data
- **Multiple Query Types**: Different views of the data (comprehensive, aggregated, matrix format)

## Project Structure

```
Provider-Cost-Analysis/
├── provider_costs_query.sql        # SQL queries for different data views
├── provider_costs_analysis.py      # Python script for analysis
├── requirements.txt                # Python dependencies
├── config_template.env             # Database configuration template
└── README.md                       # This file
```

## Prerequisites

- Python 3.8 or higher
- Database (PostgreSQL, MySQL, or SQLite)
- Database access credentials

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up database configuration:
```bash
cp config_template.env .env
# Edit .env with your actual database credentials
```

3. Ensure your database has the required tables. If not, use the sample schema provided at the bottom of `provider_costs_query.sql`.

## Database Schema

The queries assume the following tables exist:

### `providers` table
- `provider_id`: Unique identifier for the provider
- `provider_name`: Name of the provider
- `provider_type`: Type/category of provider
- `is_active`: Boolean indicating if provider is active

### `provider_costs` table
- `cost_id`: Unique identifier for the cost record
- `provider_id`: Foreign key to providers table
- `source_currency`: Source currency code (e.g., 'USD')
- `destination_currency`: Destination currency code (e.g., 'EUR')
- `cost_amount`: The cost amount
- `cost_currency`: Currency of the cost
- `cost_type`: Type of cost ('fixed', 'percentage', 'per_transaction')
- `effective_date`: Date when cost becomes effective
- `expiry_date`: Date when cost expires (NULL if no expiry)
- `is_active`: Boolean indicating if cost is active

### `exchange_rates` table (optional)
- `rate_id`: Unique identifier
- `source_currency`: Source currency code
- `destination_currency`: Destination currency code
- `exchange_rate`: Exchange rate value
- `rate_date`: Date of the exchange rate

## Usage

### Option 1: Using SQL Queries Directly

The `provider_costs_query.sql` file contains 5 different query types:

1. **Comprehensive Query**: All provider costs with full details
2. **Aggregated Query**: Costs aggregated by currency corridor
3. **Matrix Query**: Pivot view for easy comparison
4. **Corridors Query**: List of all available currency corridors
5. **Exchange Rate Context Query**: Costs with exchange rate information

You can execute these queries directly in your database client:

```sql
-- Example: Run the comprehensive query
SELECT
    p.provider_id,
    p.provider_name,
    pc.source_currency,
    pc.destination_currency,
    CONCAT(pc.source_currency, ' -> ', pc.destination_currency) AS currency_corridor,
    pc.cost_amount,
    pc.cost_type
FROM providers p
INNER JOIN provider_costs pc ON p.provider_id = pc.provider_id
WHERE pc.is_active = TRUE;
```

### Option 2: Using the Python Analysis Tool

#### Basic Usage

```python
from provider_costs_analysis import ProviderCostAnalyzer

# Configure database connection
db_config = {
    'host': 'localhost',
    'port': '5432',
    'database': 'your_database',
    'username': 'your_user',
    'password': 'your_password',
    'db_type': 'postgresql'
}

# Initialize analyzer
analyzer = ProviderCostAnalyzer(db_config)

# Fetch all data
data = analyzer.fetch_all_data()

# Display summary
analyzer.print_summary()

# Get cheapest provider for each corridor
cheapest = analyzer.get_cheapest_provider_by_corridor()
print(cheapest)

# Export results
analyzer.export_to_excel('provider_costs_analysis.xlsx')
analyzer.export_to_csv('provider_costs_export')
```

#### Using Environment Variables (Recommended for Security)

```python
import os
from dotenv import load_dotenv
from provider_costs_analysis import ProviderCostAnalyzer

# Load environment variables from .env file
load_dotenv()

db_config = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'username': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'db_type': os.getenv('DB_TYPE', 'postgresql')
}

analyzer = ProviderCostAnalyzer(db_config)
analyzer.fetch_all_data()
analyzer.export_to_excel('costs.xlsx')
```

#### Running the Script

```bash
# Update database credentials in the script first, then run:
python provider_costs_analysis.py
```

## Available Analysis Methods

The `ProviderCostAnalyzer` class provides several analysis methods:

### 1. `fetch_provider_costs(query_type)`
Fetches provider costs using different query types:
- `'comprehensive'`: All cost details
- `'aggregated'`: Aggregated by corridor
- `'corridors'`: List of currency corridors

### 2. `analyze_costs_by_corridor()`
Returns detailed statistical analysis grouped by currency corridor.

### 3. `get_cheapest_provider_by_corridor()`
Identifies the provider with the lowest cost for each corridor.

### 4. `compare_providers(corridor=None)`
Creates a comparison matrix of providers across corridors.

### 5. `export_to_excel(output_file)`
Exports all analysis results to a multi-sheet Excel file.

### 6. `export_to_csv(output_dir)`
Exports analysis results to multiple CSV files.

### 7. `get_summary_statistics()`
Returns summary statistics including:
- Total providers
- Total currency corridors
- Average, median, and range of costs
- Cost type distribution

### 8. `print_summary()`
Displays a formatted summary of the analysis.

## Output Files

### Excel Export
The Excel export creates multiple sheets:
- **comprehensive**: All provider costs with details
- **aggregated**: Aggregated costs by corridor
- **corridor_analysis**: Statistical analysis by corridor
- **cheapest_providers**: Lowest cost provider for each corridor
- **provider_comparison**: Matrix comparing providers

### CSV Export
Multiple CSV files are created:
- `provider_costs_all.csv`: All provider costs
- `corridor_summary.csv`: Summary by corridor
- `corridor_analysis.csv`: Detailed corridor analysis
- `cheapest_providers.csv`: Cheapest providers
- `provider_comparison.csv`: Provider comparison matrix

## Example Outputs

### Summary Statistics
```
======================================================================
PROVIDER COSTS SUMMARY - ALL CURRENCY CORRIDORS
======================================================================
Total Providers:        15
Total Currency Corridors: 24
Total Cost Records:     360
Active Cost Records:    342

Average Cost:           2.5432
Median Cost:            2.1000
Cost Range:             0.5000 - 8.7500

Cost Types Distribution:
  percentage: 280
  fixed: 62
  per_transaction: 18
======================================================================
```

### Cheapest Providers
```
currency_corridor  provider_name  cost_amount  cost_type  cost_currency
USD -> EUR        Provider A      1.25         percentage  USD
EUR -> USD        Provider B      1.35         percentage  EUR
USD -> GBP        Provider C      1.50         percentage  USD
```

## Customization

### Adapting to Your Schema

If your database schema differs, you'll need to update:

1. **Table Names**: Replace `providers`, `provider_costs`, `exchange_rates` with your actual table names
2. **Column Names**: Update column references to match your schema
3. **SQL Queries**: Modify the queries in both the SQL file and Python script

### Adding Custom Queries

To add custom queries, update the `queries` dictionary in the `fetch_provider_costs` method:

```python
queries = {
    'custom': """
        SELECT
            -- Your custom query here
        FROM
            your_tables
    """
}
```

### Adding Currency Corridors to Matrix View

Edit Query 3 in `provider_costs_query.sql` to add more currency pairs:

```sql
MAX(CASE WHEN CONCAT(pc.source_currency, '-', pc.destination_currency) = 'USD-JPY'
    THEN pc.cost_amount END) AS "USD_to_JPY"
```

## Security Best Practices

1. **Never commit credentials**: Add `.env` to your `.gitignore` file
2. **Use environment variables**: Store database credentials in environment variables
3. **Restrict database permissions**: Use read-only credentials if only querying
4. **Encrypt connections**: Use SSL/TLS for database connections in production

## Troubleshooting

### Connection Issues
- Verify database credentials in your config
- Check if database server is running
- Ensure firewall allows connections to database port

### Query Errors
- Verify table and column names match your schema
- Check if required tables exist
- Ensure you have necessary database permissions

### Import Errors
- Run `pip install -r requirements.txt` to install all dependencies
- Ensure you're using Python 3.8 or higher

## Support

For issues or questions:
1. Check that your database schema matches the expected structure
2. Review the sample queries in `provider_costs_query.sql`
3. Verify database connection settings

## License

This project is provided as-is for analysis purposes.

## Version History

- **v1.0.0** (2025-10-28): Initial release
  - SQL queries for provider costs
  - Python analysis tool
  - Export to Excel and CSV
  - Summary statistics and comparisons
