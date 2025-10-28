"""
Provider Costs Analysis Across Currency Corridors
==================================================
This script queries provider costs across all currency corridors and provides
comprehensive analysis and visualization capabilities.

Requirements:
    pip install pandas sqlalchemy psycopg2-binary mysql-connector-python openpyxl matplotlib seaborn

Author: Data Science Team
Date: 2025-10-28
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
import os
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class ProviderCostAnalyzer:
    """
    A class to analyze provider costs across different currency corridors.
    """

    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize the analyzer with database configuration.

        Args:
            db_config: Dictionary containing database connection parameters
                      {
                          'host': 'localhost',
                          'port': '5432',
                          'database': 'your_db',
                          'username': 'your_user',
                          'password': 'your_password',
                          'db_type': 'postgresql'  # or 'mysql', 'sqlite'
                      }
        """
        self.db_config = db_config
        self.engine = self._create_engine()
        self.provider_costs_df = None
        self.corridor_summary_df = None

    def _create_engine(self):
        """Create SQLAlchemy engine based on database type."""
        db_type = self.db_config.get('db_type', 'postgresql')

        if db_type == 'postgresql':
            connection_string = (
                f"postgresql://{self.db_config['username']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
        elif db_type == 'mysql':
            connection_string = (
                f"mysql+mysqlconnector://{self.db_config['username']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
        elif db_type == 'sqlite':
            connection_string = f"sqlite:///{self.db_config['database']}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        return create_engine(connection_string)

    def fetch_provider_costs(self, query_type: str = 'comprehensive') -> pd.DataFrame:
        """
        Fetch provider costs data from the database.

        Args:
            query_type: Type of query to execute
                       'comprehensive' - All provider costs with details
                       'aggregated' - Aggregated costs by corridor
                       'matrix' - Pivot view of costs
                       'corridors' - List of all currency corridors
                       'custom' - Use custom SQL query

        Returns:
            DataFrame containing the query results
        """
        queries = {
            'comprehensive': """
                SELECT
                    p.provider_id,
                    p.provider_name,
                    pc.source_currency,
                    pc.destination_currency,
                    CONCAT(pc.source_currency, ' -> ', pc.destination_currency) AS currency_corridor,
                    pc.cost_amount,
                    pc.cost_currency,
                    pc.cost_type,
                    pc.effective_date,
                    pc.expiry_date,
                    CASE
                        WHEN pc.expiry_date IS NULL OR pc.expiry_date > CURRENT_DATE
                        THEN 'Active'
                        ELSE 'Expired'
                    END AS status,
                    CASE
                        WHEN pc.cost_type = 'percentage' THEN pc.cost_amount
                        ELSE NULL
                    END AS percentage_cost,
                    CASE
                        WHEN pc.cost_type = 'fixed' THEN pc.cost_amount
                        ELSE NULL
                    END AS fixed_cost
                FROM
                    providers p
                INNER JOIN
                    provider_costs pc ON p.provider_id = pc.provider_id
                WHERE
                    pc.is_active = TRUE
                ORDER BY
                    p.provider_name,
                    pc.source_currency,
                    pc.destination_currency,
                    pc.effective_date DESC
            """,

            'aggregated': """
                SELECT
                    CONCAT(pc.source_currency, ' -> ', pc.destination_currency) AS currency_corridor,
                    pc.source_currency,
                    pc.destination_currency,
                    COUNT(DISTINCT p.provider_id) AS number_of_providers,
                    AVG(CASE WHEN pc.cost_type = 'percentage' THEN pc.cost_amount END) AS avg_percentage_cost,
                    MIN(CASE WHEN pc.cost_type = 'percentage' THEN pc.cost_amount END) AS min_percentage_cost,
                    MAX(CASE WHEN pc.cost_type = 'percentage' THEN pc.cost_amount END) AS max_percentage_cost,
                    AVG(CASE WHEN pc.cost_type = 'fixed' THEN pc.cost_amount END) AS avg_fixed_cost,
                    MIN(CASE WHEN pc.cost_type = 'fixed' THEN pc.cost_amount END) AS min_fixed_cost,
                    MAX(CASE WHEN pc.cost_type = 'fixed' THEN pc.cost_amount END) AS max_fixed_cost
                FROM
                    providers p
                INNER JOIN
                    provider_costs pc ON p.provider_id = pc.provider_id
                WHERE
                    pc.is_active = TRUE
                    AND (pc.expiry_date IS NULL OR pc.expiry_date > CURRENT_DATE)
                GROUP BY
                    pc.source_currency,
                    pc.destination_currency
                ORDER BY
                    pc.source_currency,
                    pc.destination_currency
            """,

            'corridors': """
                SELECT DISTINCT
                    pc.source_currency,
                    pc.destination_currency,
                    CONCAT(pc.source_currency, ' -> ', pc.destination_currency) AS currency_corridor,
                    COUNT(DISTINCT p.provider_id) AS provider_count
                FROM
                    provider_costs pc
                INNER JOIN
                    providers p ON p.provider_id = pc.provider_id
                WHERE
                    pc.is_active = TRUE
                GROUP BY
                    pc.source_currency,
                    pc.destination_currency
                ORDER BY
                    pc.source_currency,
                    pc.destination_currency
            """
        }

        query = queries.get(query_type)
        if not query:
            raise ValueError(f"Invalid query type: {query_type}")

        with self.engine.connect() as connection:
            df = pd.read_sql(text(query), connection)

        return df

    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all provider cost data using different query types.

        Returns:
            Dictionary containing DataFrames for different views
        """
        results = {}

        print("Fetching comprehensive provider costs...")
        results['comprehensive'] = self.fetch_provider_costs('comprehensive')
        self.provider_costs_df = results['comprehensive']

        print("Fetching aggregated costs by corridor...")
        results['aggregated'] = self.fetch_provider_costs('aggregated')
        self.corridor_summary_df = results['aggregated']

        print("Fetching currency corridors...")
        results['corridors'] = self.fetch_provider_costs('corridors')

        return results

    def analyze_costs_by_corridor(self) -> pd.DataFrame:
        """
        Analyze provider costs grouped by currency corridor.

        Returns:
            DataFrame with cost analysis by corridor
        """
        if self.provider_costs_df is None:
            self.provider_costs_df = self.fetch_provider_costs('comprehensive')

        analysis = self.provider_costs_df.groupby(['currency_corridor', 'source_currency', 'destination_currency']).agg({
            'provider_name': 'count',
            'cost_amount': ['mean', 'min', 'max', 'std'],
            'percentage_cost': ['mean', 'min', 'max'],
            'fixed_cost': ['mean', 'min', 'max']
        }).round(4)

        analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]
        analysis = analysis.reset_index()
        analysis.rename(columns={'provider_name_count': 'number_of_providers'}, inplace=True)

        return analysis

    def get_cheapest_provider_by_corridor(self) -> pd.DataFrame:
        """
        Find the cheapest provider for each currency corridor.

        Returns:
            DataFrame with cheapest provider for each corridor
        """
        if self.provider_costs_df is None:
            self.provider_costs_df = self.fetch_provider_costs('comprehensive')

        # Filter active providers only
        active_df = self.provider_costs_df[self.provider_costs_df['status'] == 'Active'].copy()

        # Get cheapest provider by corridor
        cheapest = active_df.loc[active_df.groupby('currency_corridor')['cost_amount'].idxmin()]

        return cheapest[['currency_corridor', 'provider_name', 'cost_amount', 'cost_type', 'cost_currency']]

    def compare_providers(self, corridor: Optional[str] = None) -> pd.DataFrame:
        """
        Compare providers for a specific corridor or all corridors.

        Args:
            corridor: Currency corridor (e.g., 'USD -> EUR'), None for all corridors

        Returns:
            DataFrame comparing providers
        """
        if self.provider_costs_df is None:
            self.provider_costs_df = self.fetch_provider_costs('comprehensive')

        df = self.provider_costs_df[self.provider_costs_df['status'] == 'Active'].copy()

        if corridor:
            df = df[df['currency_corridor'] == corridor]

        comparison = df.pivot_table(
            index='provider_name',
            columns='currency_corridor',
            values='cost_amount',
            aggfunc='first'
        ).fillna('-')

        return comparison

    def export_to_excel(self, output_file: str = 'provider_costs_analysis.xlsx'):
        """
        Export all analysis results to an Excel file with multiple sheets.

        Args:
            output_file: Path to output Excel file
        """
        if self.provider_costs_df is None:
            data = self.fetch_all_data()
        else:
            data = {
                'comprehensive': self.provider_costs_df,
                'aggregated': self.corridor_summary_df
            }

        # Additional analysis
        data['corridor_analysis'] = self.analyze_costs_by_corridor()
        data['cheapest_providers'] = self.get_cheapest_provider_by_corridor()
        data['provider_comparison'] = self.compare_providers()

        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Analysis exported to {output_file}")

    def export_to_csv(self, output_dir: str = 'provider_costs_export'):
        """
        Export analysis results to multiple CSV files.

        Args:
            output_dir: Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.provider_costs_df is None:
            self.fetch_all_data()

        # Export various datasets
        exports = {
            'provider_costs_all.csv': self.provider_costs_df,
            'corridor_summary.csv': self.corridor_summary_df,
            'corridor_analysis.csv': self.analyze_costs_by_corridor(),
            'cheapest_providers.csv': self.get_cheapest_provider_by_corridor(),
            'provider_comparison.csv': self.compare_providers()
        }

        for filename, df in exports.items():
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Exported: {filepath}")

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for the provider costs.

        Returns:
            Dictionary containing summary statistics
        """
        if self.provider_costs_df is None:
            self.provider_costs_df = self.fetch_provider_costs('comprehensive')

        stats = {
            'total_providers': self.provider_costs_df['provider_name'].nunique(),
            'total_corridors': self.provider_costs_df['currency_corridor'].nunique(),
            'total_cost_records': len(self.provider_costs_df),
            'active_costs': len(self.provider_costs_df[self.provider_costs_df['status'] == 'Active']),
            'average_cost': self.provider_costs_df['cost_amount'].mean(),
            'median_cost': self.provider_costs_df['cost_amount'].median(),
            'cost_range': (self.provider_costs_df['cost_amount'].min(), self.provider_costs_df['cost_amount'].max()),
            'cost_types': self.provider_costs_df['cost_type'].value_counts().to_dict()
        }

        return stats

    def print_summary(self):
        """Print a summary of the provider costs analysis."""
        stats = self.get_summary_statistics()

        print("\n" + "="*70)
        print("PROVIDER COSTS SUMMARY - ALL CURRENCY CORRIDORS")
        print("="*70)
        print(f"Total Providers:        {stats['total_providers']}")
        print(f"Total Currency Corridors: {stats['total_corridors']}")
        print(f"Total Cost Records:     {stats['total_cost_records']}")
        print(f"Active Cost Records:    {stats['active_costs']}")
        print(f"\nAverage Cost:           {stats['average_cost']:.4f}")
        print(f"Median Cost:            {stats['median_cost']:.4f}")
        print(f"Cost Range:             {stats['cost_range'][0]:.4f} - {stats['cost_range'][1]:.4f}")
        print(f"\nCost Types Distribution:")
        for cost_type, count in stats['cost_types'].items():
            print(f"  {cost_type}: {count}")
        print("="*70 + "\n")


def main():
    """
    Main execution function with example usage.
    """
    # Database configuration
    # IMPORTANT: Update these values with your actual database credentials
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'your_database_name',
        'username': 'your_username',
        'password': 'your_password',
        'db_type': 'postgresql'  # Options: 'postgresql', 'mysql', 'sqlite'
    }

    # Alternative: Load from environment variables for security
    # db_config = {
    #     'host': os.getenv('DB_HOST', 'localhost'),
    #     'port': os.getenv('DB_PORT', '5432'),
    #     'database': os.getenv('DB_NAME'),
    #     'username': os.getenv('DB_USER'),
    #     'password': os.getenv('DB_PASSWORD'),
    #     'db_type': os.getenv('DB_TYPE', 'postgresql')
    # }

    try:
        # Initialize analyzer
        print("Initializing Provider Cost Analyzer...")
        analyzer = ProviderCostAnalyzer(db_config)

        # Fetch all data
        print("\nFetching data from database...")
        all_data = analyzer.fetch_all_data()

        # Display summary
        analyzer.print_summary()

        # Display sample data
        print("\nSample Provider Costs Data (First 10 rows):")
        print(analyzer.provider_costs_df.head(10))

        print("\n\nCurrency Corridors Summary:")
        print(analyzer.corridor_summary_df)

        # Get cheapest providers
        print("\n\nCheapest Provider by Corridor:")
        cheapest = analyzer.get_cheapest_provider_by_corridor()
        print(cheapest)

        # Corridor analysis
        print("\n\nDetailed Corridor Analysis:")
        corridor_analysis = analyzer.analyze_costs_by_corridor()
        print(corridor_analysis)

        # Provider comparison
        print("\n\nProvider Comparison Matrix:")
        comparison = analyzer.compare_providers()
        print(comparison)

        # Export results
        print("\n\nExporting results...")
        analyzer.export_to_excel('provider_costs_analysis.xlsx')
        analyzer.export_to_csv('provider_costs_export')

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
