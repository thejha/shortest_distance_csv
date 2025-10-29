#!/usr/bin/env python3
"""
Algorithm Analytics Viewer
==========================

This script displays analytics from the model_comparison_results table
to help monitor algorithm selection patterns and performance.

Usage: python view_algorithm_analytics.py
"""

import sys
import mysql.connector
from datetime import datetime
import pandas as pd

# Database configuration - using the same config as the worker
def get_mysql_connection():
    """Create a connection to the MySQL database."""
    return mysql.connector.connect(
        host='vaibhavjha.mysql.pythonanywhere-services.com',
        user='vaibhavjha',
        password='anubhav21',
        database='vaibhavjha$database'
    )

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*80}")
    print(f"🔍 {title.upper()}")
    print(f"{'='*80}")

def print_section(title):
    """Print a formatted section header."""
    print(f"\n📊 {title}")
    print("-" * 60)

def get_algorithm_frequency():
    """Get frequency of algorithm selection."""
    query = """
    SELECT selected_model, COUNT(*) as frequency
    FROM model_comparison_results
    GROUP BY selected_model
    ORDER BY frequency DESC;
    """

    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        print_section("Algorithm Selection Frequency")

        if results:
            total_jobs = sum(row[1] for row in results)
            print(f"{'Algorithm':<15} {'Count':<10} {'Percentage':<12} {'Visual':<20}")
            print("-" * 60)

            for algorithm, count in results:
                percentage = (count / total_jobs) * 100
                bar_length = int(percentage / 5)  # Scale bar to 20 chars max
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"{algorithm:<15} {count:<10} {percentage:>6.1f}%     {bar}")

            print(f"\nTotal Jobs Processed: {total_jobs}")
        else:
            print("No algorithm selection data found.")

    except Exception as e:
        print(f"❌ Error getting algorithm frequency: {e}")

def get_cost_performance_by_hub():
    """Get average cost performance by hub."""
    query = """
    SELECT hub_id,
           AVG(hdbscan_cost_per_kg) as avg_hdbscan_cost,
           AVG(zscore_cost_per_kg) as avg_zscore_cost,
           COUNT(*) as job_count
    FROM model_comparison_results
    GROUP BY hub_id
    ORDER BY hub_id;
    """

    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        print_section("Average Cost Performance by Hub")

        if results:
            print(f"{'Hub ID':<8} {'HDBSCAN ₹/kg':<15} {'Z-Score ₹/kg':<15} {'Better':<10} {'Jobs':<6} {'Savings':<10}")
            print("-" * 80)

            for hub_id, hdbscan_cost, zscore_cost, job_count in results:
                if hdbscan_cost and zscore_cost:
                    better = "HDBSCAN" if hdbscan_cost < zscore_cost else "Z-Score"
                    savings = abs(hdbscan_cost - zscore_cost)
                    print(f"{hub_id:<8} {hdbscan_cost:>11.2f}     {zscore_cost:>11.2f}     {better:<10} {job_count:<6} ₹{savings:>5.2f}")
                else:
                    print(f"{hub_id:<8} {'N/A':<15} {'N/A':<15} {'N/A':<10} {job_count:<6} {'N/A':<10}")
        else:
            print("No cost performance data found.")

    except Exception as e:
        print(f"❌ Error getting cost performance data: {e}")

def get_recent_selections():
    """Get recent algorithm selections."""
    query = """
    SELECT hub_id, selected_model, selection_reason, created_at
    FROM model_comparison_results
    ORDER BY created_at DESC
    LIMIT 10;
    """

    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        print_section("Recent Algorithm Selections (Last 10)")

        if results:
            print(f"{'Hub':<6} {'Algorithm':<10} {'Reason':<40} {'Date':<20}")
            print("-" * 80)

            for hub_id, algorithm, reason, created_at in results:
                # Truncate long reasons
                short_reason = reason[:37] + "..." if len(reason) > 40 else reason
                date_str = created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else "N/A"
                print(f"{hub_id:<6} {algorithm:<10} {short_reason:<40} {date_str}")
        else:
            print("No recent selection data found.")

    except Exception as e:
        print(f"❌ Error getting recent selections: {e}")

def get_overall_statistics():
    """Get overall statistics."""
    queries = {
        'total_jobs': "SELECT COUNT(*) FROM model_comparison_results",
        'hdbscan_wins': "SELECT COUNT(*) FROM model_comparison_results WHERE selected_model = 'HDBSCAN'",
        'zscore_wins': "SELECT COUNT(*) FROM model_comparison_results WHERE selected_model = 'Z-Score'",
        'avg_cost_improvement': "SELECT AVG(ABS(cost_improvement_percent)) FROM model_comparison_results",
        'avg_processing_time_hdbscan': "SELECT AVG(hdbscan_processing_time) FROM model_comparison_results",
        'avg_processing_time_zscore': "SELECT AVG(zscore_processing_time) FROM model_comparison_results",
        'total_weight_processed': "SELECT SUM(total_weight_kg) FROM model_comparison_results",
        'unique_hubs': "SELECT COUNT(DISTINCT hub_id) FROM model_comparison_results"
    }

    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()

        stats = {}
        for key, query in queries.items():
            cursor.execute(query)
            result = cursor.fetchone()
            stats[key] = result[0] if result and result[0] is not None else 0

        conn.close()

        print_section("Overall Performance Statistics")

        total_jobs = stats['total_jobs']
        if total_jobs > 0:
            hdbscan_percentage = (stats['hdbscan_wins'] / total_jobs) * 100
            zscore_percentage = (stats['zscore_wins'] / total_jobs) * 100

            print(f"📈 Total Jobs Processed: {total_jobs:,}")
            print(f"🏢 Unique Hubs Analyzed: {stats['unique_hubs']:,}")
            print(f"📦 Total Weight Processed: {stats['total_weight_processed']:,.0f} kg")
            print()
            print(f"🎯 Algorithm Selection:")
            print(f"   • HDBSCAN Selected: {stats['hdbscan_wins']:,} times ({hdbscan_percentage:.1f}%)")
            print(f"   • Z-Score Selected: {stats['zscore_wins']:,} times ({zscore_percentage:.1f}%)")
            print()
            print(f"💰 Average Cost Impact: {stats['avg_cost_improvement']:.1f}%")
            print()
            print(f"⏱️  Processing Time:")
            print(f"   • HDBSCAN Average: {stats['avg_processing_time_hdbscan']:.1f} seconds")
            print(f"   • Z-Score Average: {stats['avg_processing_time_zscore']:.1f} seconds")
        else:
            print("No statistics available - no jobs processed yet.")

    except Exception as e:
        print(f"❌ Error getting overall statistics: {e}")

def get_cost_savings_analysis():
    """Analyze actual cost savings from algorithm selection."""
    query = """
    SELECT
        hub_id,
        hdbscan_cost_per_kg,
        zscore_cost_per_kg,
        selected_model,
        total_weight_kg,
        (CASE
            WHEN selected_model = 'HDBSCAN' THEN (zscore_cost_per_kg - hdbscan_cost_per_kg) * total_weight_kg
            WHEN selected_model = 'Z-Score' THEN (hdbscan_cost_per_kg - zscore_cost_per_kg) * total_weight_kg
            ELSE 0
        END) as actual_savings,
        created_at
    FROM model_comparison_results
    WHERE hdbscan_cost_per_kg IS NOT NULL AND zscore_cost_per_kg IS NOT NULL
    ORDER BY actual_savings DESC
    LIMIT 15;
    """

    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        print_section("Top Cost Savings Achievements")

        if results:
            print(f"{'Hub':<6} {'Selected':<10} {'Savings ₹':<12} {'Weight kg':<10} {'Date':<12}")
            print("-" * 60)

            total_savings = 0
            for hub_id, hdbscan_cost, zscore_cost, selected, weight, savings, created_at in results:
                if savings and savings > 0:
                    total_savings += savings
                    date_str = created_at.strftime("%Y-%m-%d") if created_at else "N/A"
                    print(f"{hub_id:<6} {selected:<10} {savings:>8.0f}     {weight:>7.0f}    {date_str}")

            print(f"\n💰 Total Estimated Savings: ₹{total_savings:,.0f}")
        else:
            print("No cost savings data available.")

    except Exception as e:
        print(f"❌ Error getting cost savings analysis: {e}")

def check_table_exists():
    """Check if the model_comparison_results table exists."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES LIKE 'model_comparison_results'")
        result = cursor.fetchone()
        conn.close()
        return result is not None
    except Exception as e:
        print(f"❌ Error checking table existence: {e}")
        return False

def main():
    """Main function to display all analytics."""
    print("🚀 INTELLIGENT OPTIMIZATION WORKER - ANALYTICS DASHBOARD")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if table exists
    if not check_table_exists():
        print("\n❌ Error: model_comparison_results table does not exist.")
        print("Please run the worker at least once to create the table and generate data.")
        print("\nTo create the table manually, run:")
        print("mysql> SOURCE create_model_comparison_results_table.sql;")
        return

    try:
        # Display all analytics
        get_overall_statistics()
        get_algorithm_frequency()
        get_cost_performance_by_hub()
        get_cost_savings_analysis()
        get_recent_selections()

        print_header("analytics dashboard complete")
        print("🔄 Run this script anytime to see updated analytics")
        print("📊 Data refreshes automatically as the worker processes more jobs")

    except KeyboardInterrupt:
        print("\n\n👋 Analytics viewer stopped by user")
    except Exception as e:
        print(f"\n❌ Error displaying analytics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
