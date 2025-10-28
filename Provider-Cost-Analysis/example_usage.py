"""
Example Usage: Provider Cost Analysis
======================================
This script demonstrates various ways to use the ProviderCostAnalyzer class.

Run this after setting up your database configuration.
"""

import os
from provider_costs_analysis import ProviderCostAnalyzer
from dotenv import load_dotenv


def example_1_basic_analysis():
    """
    Example 1: Basic analysis with direct database configuration
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Analysis")
    print("="*70)

    # Direct configuration (not recommended for production)
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'your_database',
        'username': 'your_user',
        'password': 'your_password',
        'db_type': 'postgresql'
    }

    analyzer = ProviderCostAnalyzer(db_config)
    analyzer.fetch_all_data()
    analyzer.print_summary()


def example_2_environment_variables():
    """
    Example 2: Using environment variables (recommended)
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Using Environment Variables")
    print("="*70)

    # Load from .env file
    load_dotenv()

    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'username': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'db_type': os.getenv('DB_TYPE', 'postgresql')
    }

    analyzer = ProviderCostAnalyzer(db_config)

    # Fetch comprehensive data
    print("\nFetching comprehensive provider costs...")
    comprehensive_df = analyzer.fetch_provider_costs('comprehensive')
    print(f"Fetched {len(comprehensive_df)} records")
    print("\nFirst 5 records:")
    print(comprehensive_df.head())


def example_3_corridor_analysis():
    """
    Example 3: Analyze costs by currency corridor
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Corridor Analysis")
    print("="*70)

    load_dotenv()
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'username': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'db_type': os.getenv('DB_TYPE', 'postgresql')
    }

    analyzer = ProviderCostAnalyzer(db_config)
    analyzer.fetch_all_data()

    # Get corridor analysis
    print("\nAnalyzing costs by corridor...")
    corridor_analysis = analyzer.analyze_costs_by_corridor()
    print(corridor_analysis)

    # Get all available corridors
    print("\n\nAvailable Currency Corridors:")
    corridors_df = analyzer.fetch_provider_costs('corridors')
    print(corridors_df)


def example_4_find_best_providers():
    """
    Example 4: Find the cheapest provider for each corridor
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Find Best Providers")
    print("="*70)

    load_dotenv()
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'username': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'db_type': os.getenv('DB_TYPE', 'postgresql')
    }

    analyzer = ProviderCostAnalyzer(db_config)
    analyzer.fetch_all_data()

    # Get cheapest providers
    print("\nCheapest provider for each corridor:")
    cheapest = analyzer.get_cheapest_provider_by_corridor()
    print(cheapest)

    # Compare all providers
    print("\n\nProvider comparison matrix:")
    comparison = analyzer.compare_providers()
    print(comparison)


def example_5_specific_corridor():
    """
    Example 5: Analyze a specific currency corridor
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Analyze Specific Corridor")
    print("="*70)

    load_dotenv()
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'username': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'db_type': os.getenv('DB_TYPE', 'postgresql')
    }

    analyzer = ProviderCostAnalyzer(db_config)
    analyzer.fetch_all_data()

    # Analyze specific corridor (e.g., USD -> EUR)
    corridor = 'USD -> EUR'
    print(f"\nAnalyzing corridor: {corridor}")

    if analyzer.provider_costs_df is not None:
        corridor_data = analyzer.provider_costs_df[
            analyzer.provider_costs_df['currency_corridor'] == corridor
        ]

        if not corridor_data.empty:
            print(f"\nFound {len(corridor_data)} providers for {corridor}")
            print(corridor_data[['provider_name', 'cost_amount', 'cost_type', 'status']])

            # Statistics for this corridor
            print(f"\n{corridor} Statistics:")
            print(f"Average cost: {corridor_data['cost_amount'].mean():.4f}")
            print(f"Minimum cost: {corridor_data['cost_amount'].min():.4f}")
            print(f"Maximum cost: {corridor_data['cost_amount'].max():.4f}")
        else:
            print(f"No data found for corridor: {corridor}")


def example_6_export_results():
    """
    Example 6: Export results to Excel and CSV
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Export Results")
    print("="*70)

    load_dotenv()
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'username': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'db_type': os.getenv('DB_TYPE', 'postgresql')
    }

    analyzer = ProviderCostAnalyzer(db_config)
    analyzer.fetch_all_data()

    # Export to Excel
    print("\nExporting to Excel...")
    analyzer.export_to_excel('provider_costs_report.xlsx')

    # Export to CSV
    print("\nExporting to CSV files...")
    analyzer.export_to_csv('provider_costs_csv_export')

    print("\nExport complete!")


def example_7_custom_filters():
    """
    Example 7: Apply custom filters to the data
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Custom Filters")
    print("="*70)

    load_dotenv()
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'username': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'db_type': os.getenv('DB_TYPE', 'postgresql')
    }

    analyzer = ProviderCostAnalyzer(db_config)
    analyzer.fetch_all_data()

    if analyzer.provider_costs_df is not None:
        df = analyzer.provider_costs_df

        # Filter 1: Only percentage-based costs
        print("\n1. Percentage-based costs:")
        percentage_costs = df[df['cost_type'] == 'percentage']
        print(f"Found {len(percentage_costs)} percentage-based costs")
        print(percentage_costs[['provider_name', 'currency_corridor', 'cost_amount']].head(10))

        # Filter 2: Costs for specific source currency (e.g., USD)
        print("\n\n2. Costs for USD as source currency:")
        usd_costs = df[df['source_currency'] == 'USD']
        print(f"Found {len(usd_costs)} USD source currency records")
        print(usd_costs[['provider_name', 'destination_currency', 'cost_amount']].head(10))

        # Filter 3: Active costs only
        print("\n\n3. Active costs only:")
        active_costs = df[df['status'] == 'Active']
        print(f"Found {len(active_costs)} active cost records")

        # Filter 4: Costs below threshold
        threshold = 2.0
        print(f"\n\n4. Costs below {threshold}:")
        low_costs = df[df['cost_amount'] < threshold]
        print(f"Found {len(low_costs)} costs below {threshold}")
        print(low_costs[['provider_name', 'currency_corridor', 'cost_amount']].head(10))


def main():
    """
    Main function to run all examples
    """
    print("\n" + "="*70)
    print("PROVIDER COST ANALYSIS - EXAMPLE USAGE")
    print("="*70)
    print("\nThis script demonstrates various ways to use the analyzer.")
    print("Make sure you have:")
    print("1. Set up your .env file with database credentials")
    print("2. Installed required packages: pip install -r requirements.txt")
    print("3. Database with provider_costs and providers tables")
    print("\nChoose an example to run:")
    print("1. Basic Analysis")
    print("2. Using Environment Variables")
    print("3. Corridor Analysis")
    print("4. Find Best Providers")
    print("5. Analyze Specific Corridor")
    print("6. Export Results")
    print("7. Custom Filters")
    print("0. Run All Examples")

    choice = input("\nEnter your choice (0-7): ").strip()

    examples = {
        '1': example_1_basic_analysis,
        '2': example_2_environment_variables,
        '3': example_3_corridor_analysis,
        '4': example_4_find_best_providers,
        '5': example_5_specific_corridor,
        '6': example_6_export_results,
        '7': example_7_custom_filters
    }

    if choice == '0':
        # Run all examples (except basic which requires manual config)
        for key in ['2', '3', '4', '5', '6', '7']:
            try:
                examples[key]()
            except Exception as e:
                print(f"\nError in example {key}: {str(e)}")
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
