#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Distance Data Upload Script
===========================

This script uploads the distances dump CSV file to the local MySQL database
to populate the distances table with pre-calculated distance data.

Usage:
    python upload_distances.py distances_dump.csv

Features:
- Handles large CSV files efficiently
- Uses batch inserts for better performance
- Handles data type conversions properly
- Provides progress updates
- Skips duplicate entries
"""

import pandas as pd
import mysql.connector
import sys
import os
from datetime import datetime
import argparse

# ------------------------------------------------
# MySQL database configuration
# ------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "rmivuxg",
}

# ------------------------------------------------
# Database connection helper
# ------------------------------------------------
def get_mysql_connection():
    """Create a connection to the MySQL database."""
    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database="route_optimization"
    )

def create_distances_table():
    """Create the distances table if it doesn't exist."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS distances (
                id INT AUTO_INCREMENT PRIMARY KEY,
                source VARCHAR(255) NOT NULL,
                destination VARCHAR(255) NOT NULL,
                distance DECIMAL(10,2) NOT NULL,
                timestamp TIMESTAMP NULL,
                source_geocode VARCHAR(255),
                destination_geocode VARCHAR(255),
                UNIQUE KEY unique_route (source, destination),
                INDEX idx_source (source),
                INDEX idx_destination (destination)
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("Distances table created/verified successfully")
        return True
        
    except Exception as e:
        print(f"Error creating distances table: {e}")
        return False

def upload_distances_data(csv_file_path, batch_size=1000):
    """
    Upload distances data from CSV file to MySQL database.
    
    Args:
        csv_file_path: Path to the CSV file
        batch_size: Number of records to insert in each batch
    """
    print(f"Starting upload of distances data from: {csv_file_path}")
    
    try:
        # Read CSV file
        print("Reading CSV file...")
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} records from CSV")
        
        # Clean and prepare data
        print("Preparing data...")
        
        # Handle missing values
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
        
        # Handle timestamp column - convert invalid dates to None
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # Replace invalid dates (like 0001-01-01) with None
        df.loc[df['timestamp'].dt.year < 1900, 'timestamp'] = None
        
        # Replace NaN values with None for MySQL
        df = df.where(pd.notnull(df), None)
        
        # Remove rows with invalid distances
        df = df.dropna(subset=['distance'])
        print(f"After cleaning: {len(df)} valid records")
        
        # Connect to database
        conn = get_mysql_connection()
        cursor = conn.cursor()
        
        # Prepare insert statement (skip timestamp for now)
        insert_query = """
            INSERT INTO distances (source, destination, distance, source_geocode, destination_geocode)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                distance = VALUES(distance),
                source_geocode = VALUES(source_geocode),
                destination_geocode = VALUES(destination_geocode)
        """
        
        # Process data in batches
        total_records = len(df)
        uploaded_count = 0
        skipped_count = 0
        
        print(f"Starting batch upload (batch size: {batch_size})...")
        
        for i in range(0, total_records, batch_size):
            batch_df = df.iloc[i:i + batch_size]
            
            # Prepare batch data (skip timestamp)
            batch_data = []
            for _, row in batch_df.iterrows():
                batch_data.append((
                    row['source'],
                    row['destination'],
                    float(row['distance']),
                    row['source_geocode'],
                    row['destination_geocode']
                ))
            
            try:
                # Execute batch insert
                cursor.executemany(insert_query, batch_data)
                conn.commit()
                
                uploaded_count += len(batch_data)
                
                # Progress update
                progress = (i + len(batch_data)) / total_records * 100
                print(f"Progress: {progress:.1f}% ({uploaded_count}/{total_records} records uploaded)")
                
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1}: {e}")
                # Continue with next batch
                continue
        
        cursor.close()
        conn.close()
        
        print(f"\nUpload completed!")
        print(f"Total records processed: {total_records}")
        print(f"Records uploaded/updated: {uploaded_count}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_file_path}")
        return False
    except Exception as e:
        print(f"Error during upload: {e}")
        return False

def verify_upload():
    """Verify that the data was uploaded correctly."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM distances")
        total_count = cursor.fetchone()[0]
        
        # Get sample data
        cursor.execute("SELECT * FROM distances LIMIT 5")
        sample_data = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        print(f"\nVerification:")
        print(f"Total records in database: {total_count}")
        print(f"Sample records:")
        for record in sample_data:
            print(f"  {record[1]} -> {record[2]}: {record[3]} km")
        
        return True
        
    except Exception as e:
        print(f"Error during verification: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Upload distances data from CSV to MySQL database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload distances data
  python upload_distances.py distances_dump.csv
  
  # Upload with custom batch size
  python upload_distances.py distances_dump.csv --batch-size 5000
        """
    )
    
    parser.add_argument('csv_file', help='Path to the distances CSV file')
    parser.add_argument('--batch-size', type=int, default=1000, 
                       help='Number of records to insert in each batch (default: 1000)')
    parser.add_argument('--verify-only', action='store_true', 
                       help='Only verify existing data, do not upload')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    print("Distance Data Upload Script")
    print("=" * 40)
    print(f"CSV file: {args.csv_file}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Verify database connection
    print("Checking database connection...")
    try:
        conn = get_mysql_connection()
        conn.close()
        print("Database connection successful")
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Please ensure MySQL is running and credentials are correct")
        return
    
    # Create table if needed
    if not create_distances_table():
        return
    
    if args.verify_only:
        verify_upload()
        return
    
    # Upload data
    success = upload_distances_data(args.csv_file, args.batch_size)
    
    if success:
        print("\nUpload completed successfully!")
        verify_upload()
    else:
        print("\nUpload failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
