#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database Cleanup Script
=======================

This script cleans up all optimization jobs and related tables,
keeping only the distances table intact.

Usage:
    python cleanup_database.py
"""

import mysql.connector
import sys

# MySQL database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "rmivuxg",
}

def get_mysql_connection():
    """Create a connection to the MySQL database."""
    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database="route_optimization"
    )

def cleanup_database():
    """Clean up all jobs and related tables, keeping only distances."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        
        print("Starting database cleanup...")
        
        # 1. Drop optimization jobs table if it exists
        print("Dropping optimization_jobs table...")
        cursor.execute("DROP TABLE IF EXISTS optimization_jobs")
        
        # 2. Drop vehicle configs table if it exists
        print("Dropping vehicle_configs table...")
        cursor.execute("DROP TABLE IF EXISTS vehicle_configs")
        
        # 3. Drop any other job-related tables
        print("Dropping other job-related tables...")
        cursor.execute("DROP TABLE IF EXISTS route_optimization_jobs")
        cursor.execute("DROP TABLE IF EXISTS tsp_jobs")
        cursor.execute("DROP TABLE IF EXISTS route_opt_jobs")
        
        # 4. Verify distances table still exists and show count
        print("Checking distances table...")
        cursor.execute("SELECT COUNT(*) FROM distances")
        distance_count = cursor.fetchone()[0]
        print(f"Distances table intact with {distance_count:,} records")
        
        # 5. Show remaining tables
        print("\nRemaining tables in database:")
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        for table in tables:
            print(f"  - {table[0]}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\nDatabase cleanup completed successfully!")
        print("Only the distances table remains with all distance data intact.")
        
        return True
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return False

def main():
    print("Database Cleanup Script")
    print("=" * 40)
    print("This will remove all optimization jobs and related tables.")
    print("Only the distances table will remain intact.")
    print()
    print("Proceeding with cleanup...")
    
    # Test database connection
    print("Testing database connection...")
    try:
        conn = get_mysql_connection()
        conn.close()
        print("Database connection successful")
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Please check your MySQL credentials and ensure the server is running.")
        return
    
    # Perform cleanup
    success = cleanup_database()
    
    if success:
        print("\nCleanup completed successfully!")
        print("You can now start fresh with the web UI.")
    else:
        print("\nCleanup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
