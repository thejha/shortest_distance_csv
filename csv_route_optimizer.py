#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV-Based Route Optimization System
====================================

This script modifies the existing routing algorithm to work with CSV files instead of API calls.
It reads pickup data from a CSV file and outputs optimized routes to another CSV file.

Key Features:
- Reads pickup data from CSV input file
- Uses HDBSCAN clustering algorithm for spatial route optimization
- Falls back to Z-score methodology when HDBSCAN is not suitable
- Outputs optimized routes to CSV file instead of API push
- Maintains all existing vehicle and cost optimization logic
- Provides sample CSV template for user data input

Usage:
    python csv_route_optimizer.py input_file.csv output_file.csv hub_id

Required CSV Columns:
    pickup_number, material_quantity, city_name, state_name, location_pincode, hub_id
"""

# ------------------------------------------------
# Imports
# ------------------------------------------------
import pandas as pd
import mysql.connector
import requests
from itertools import combinations
import json
import uuid
from datetime import datetime
import pytz
import numpy as np
import itertools
import time
import warnings
import sys
import os
import argparse
import traceback

# Import Google Maps if available
try:
    import googlemaps
    GOOGLEMAPS_AVAILABLE = True
    print("Google Maps API available")
except ImportError:
    GOOGLEMAPS_AVAILABLE = False
    print("Google Maps not available, will use OlaMaps API only")

# Import HDBSCAN if available
try:
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    HDBSCAN_AVAILABLE = True
    print("HDBSCAN clustering available")
except ImportError as e:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not available, will use original Z-score algorithm")

# Import HDBSCAN router if available
try:
    from route_opt_worker_hdbscan import HDBSCANEnhancedRouter
    HDBSCAN_ROUTER_AVAILABLE = True
    print("HDBSCAN Enhanced Router available")
except ImportError as e:
    HDBSCAN_ROUTER_AVAILABLE = False
    print("HDBSCAN Enhanced Router not available")

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

# ------------------------------------------------
# API clients & globals
# ------------------------------------------------
if GOOGLEMAPS_AVAILABLE:
    gmaps = googlemaps.Client(key='AIzaSyDMtTPCRVzDEyv-rcwk6BNIWZi4-bI-WZo')
else:
    gmaps = None

new_api_key = 'UApDWhcNIesmLCt65ZmpWCskYvEWUg1PxqwCaXhn'
request_id  = str(uuid.uuid4())
BATCH_SIZE  = 10
IST_ZONE    = pytz.timezone("Asia/Kolkata")

# ------------------------------------------------
# MySQL database configuration
# ------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "rmivuxg",
}

# ------------------------------------------------
# MySQL connection helper
# ------------------------------------------------
def get_mysql_connection():
    """Create a connection to the MySQL database."""
    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database="route_optimization"
    )

def update_job_status_message(job_id, message):
    """Update the status message for a running optimization job."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE optimization_jobs 
            SET status_message = %s
            WHERE job_id = %s
        """, (message, job_id))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[STATUS] Updated job {job_id}: {message}")
    except Exception as e:
        print(f"Warning: Could not update status message: {e}")

def update_job_status(job_id, status, output_file=None, error_message=None):
    """Update the job status to COMPLETED, ERROR, etc."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        
        # Normalize to backend enum 'COMPLETE'
        if status in ('COMPLETED', 'COMPLETE'):
            cursor.execute("""
                UPDATE optimization_jobs 
                SET status = %s, status_message = %s, output_file = %s, finished_at = NOW()
                WHERE job_id = %s
            """, ('COMPLETE', 'Route optimization completed successfully!', output_file, job_id))
        elif status == 'ERROR':
            cursor.execute("""
                UPDATE optimization_jobs 
                SET status = %s, status_message = %s, error_message = %s, finished_at = NOW()
                WHERE job_id = %s
            """, (status, 'Route optimization failed', error_message, job_id))
        else:
            cursor.execute("""
                UPDATE optimization_jobs 
                SET status = %s
                WHERE job_id = %s
            """, (status, job_id))
            
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[STATUS] Updated job {job_id} to {'COMPLETE' if status in ('COMPLETED','COMPLETE') else status}")
    except Exception as e:
        print(f"Warning: Could not update job status: {e}")

def create_database_and_tables():
    """Create the database and necessary tables if they don't exist."""
    try:
        # First connect without specifying database
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS route_optimization")
        cursor.execute("USE route_optimization")
        
        # Create distances table for caching
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS distances (
                id INT AUTO_INCREMENT PRIMARY KEY,
                source VARCHAR(255) NOT NULL,
                destination VARCHAR(255) NOT NULL,
                distance DECIMAL(10,2) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_geocode VARCHAR(255),
                destination_geocode VARCHAR(255),
                UNIQUE KEY unique_route (source, destination),
                INDEX idx_source (source),
                INDEX idx_destination (destination)
            )
        """)
        
        # Create route_optimization_jobs table for tracking jobs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS route_optimization_jobs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                hub_id INT NOT NULL,
                status ENUM('PENDING', 'RUNNING', 'COMPLETED', 'ERROR') DEFAULT 'PENDING',
                input_file VARCHAR(500),
                output_file VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP NULL,
                finished_at TIMESTAMP NULL,
                output_json LONGTEXT,
                error_message TEXT,
                INDEX idx_status (status),
                INDEX idx_hub_id (hub_id)
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("Database and tables created successfully")
        return True
        
    except Exception as e:
        print(f"Error creating database/tables: {e}")
        return False

# ------------------------------------------------
# Distance-cache helpers
# ------------------------------------------------
def get_distance_from_db(source, destination):
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT distance FROM distances WHERE source=%s AND destination=%s",
        (source, destination)
    )
    result = cursor.fetchone()
    conn.close()
    # Convert Decimal to float for consistency
    return float(result[0]) if result else None

def save_distance_to_db(source, destination, distance,
                        source_geocode=None, destination_geocode=None):
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO distances
              (source, destination, distance, timestamp,
               source_geocode, destination_geocode)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
              distance = VALUES(distance),
              timestamp = VALUES(timestamp),
              source_geocode = VALUES(source_geocode),
              destination_geocode = VALUES(destination_geocode)
        """,
        (
            source, destination, distance,
            datetime.now(IST_ZONE).strftime("%Y-%m-%d %H:%M:%S"),
            source_geocode, destination_geocode
        )
    )
    conn.commit()
    conn.close()

# ------------------------------------------------
# Google & OlaMaps utilities
# ------------------------------------------------
def batch_geocode(addresses):
    if not GOOGLEMAPS_AVAILABLE:
        print("Google Maps not available, skipping geocoding")
        return {}, addresses  # Return empty geocodes and all addresses as failed
    
    geocodes, failed = {}, []
    for i in range(0, len(addresses), BATCH_SIZE):
        batch = addresses[i:i + BATCH_SIZE]
        try:
            response = gmaps.geocode(batch)
            for addr, res in zip(batch, response):
                if res and res['geometry']:
                    loc = res['geometry']['location']
                    geocodes[addr] = f"{loc['lat']},{loc['lng']}"
                else:
                    failed.append(addr)
        except Exception as e:
            print(f"Geocode batch error: {e}")
            failed.extend(batch)
    return geocodes, failed

def batch_calculate_distances(origins, destinations):
    if not GOOGLEMAPS_AVAILABLE:
        print("Google Maps not available, skipping distance calculation")
        return {}, [(o, d) for o in origins for d in destinations]
    
    print(f"[DISTANCE] Calculating distances for {len(origins)} origins x {len(destinations)} destinations = {len(origins) * len(destinations)} pairs")
    
    dists, failed = {}, []
    total_batches = (len(origins) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(origins), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        obatch = origins[i:i + BATCH_SIZE]
        dbatch = destinations[i:i + BATCH_SIZE]
        print(f"[DISTANCE] Processing batch {batch_num}/{total_batches}: {len(obatch)} origins x {len(destinations)} destinations")
        try:
            resp = gmaps.distance_matrix(obatch, dbatch)
            for o, d, row in zip(obatch, dbatch, resp['rows']):
                elem = row['elements'][0]
                if elem['status'] == 'OK':
                    distance_km = elem['distance']['value'] / 1000.0
                    dists[(o, d)] = distance_km
                    print(f"[DISTANCE] {o} -> {d}: {distance_km:.2f} km")
                else:
                    failed.append((o, d))
                    print(f"[DISTANCE] Failed: {o} -> {d} (status: {elem['status']})")
        except Exception as e:
            print(f"[DISTANCE] Batch error: {e}")
            failed.extend([(o, d) for o in obatch for d in dbatch])
    
    print(f"[DISTANCE] Completed: {len(dists)} successful, {len(failed)} failed")
    return dists, failed

def calculate_dist(source, destination):
    # 1) DB cache
    dist = get_distance_from_db(source, destination)
    if dist is not None:
        print(f"[DISTANCE] From cache: {source} -> {destination}: {dist:.2f} km")
        return dist

    # 2) Google
    print(f"[DISTANCE] Calculating: {source} -> {destination}")
    dists, _ = batch_calculate_distances([source], [destination])
    if (source, destination) in dists:
        dist = dists[(source, destination)]
        save_distance_to_db(source, destination, dist)
        print(f"[DISTANCE] Calculated: {source} -> {destination}: {dist:.2f} km")
        return dist

    # 3) OlaMaps fallback
    print(f"[DISTANCE] Trying OlaMaps fallback: {source} -> {destination}")
    return calculate_dist_new_api(source, destination)

def calculate_dist_new_api(source, destination):
    sg = get_geocode(source)
    dg = get_geocode(destination)
    if not sg or not dg:
        print(f"[DISTANCE] Failed geocode -> OlaMaps skipped: {source} -> {destination}")
        return None

    url = (f"https://api.olamaps.io/routing/v1/distanceMatrix?"
           f"origins={sg}&destinations={dg}&api_key={new_api_key}")
    print(f"[DISTANCE] OlaMaps API call: {source} -> {destination}")
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        elem = data['rows'][0]['elements'][0]
        if elem['status'] == 'OK':
            km = elem['distance'] / 1000.0
            save_distance_to_db(source, destination, km, sg, dg)
            print(f"[DISTANCE] OlaMaps success: {source} -> {destination}: {km:.2f} km")
            return km
        print(f"[DISTANCE] OlaMaps element error: {elem['status']} for {source} -> {destination}")
    else:
        print(f"[DISTANCE] OlaMaps HTTP {resp.status_code} for {source} -> {destination}")
    return None

def get_geocode(address):
    gcs, _ = batch_geocode([address])
    return gcs.get(address)

# ------------------------------------------------
# JSON-safe NumPy converter
# ------------------------------------------------
def convert_np(obj):
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_np(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj

# ------------------------------------------------
# Database vehicle loading
# ------------------------------------------------
def load_vehicles_from_db(hub_id):
    """Load vehicle configurations from database for the specified hub_id."""
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root', 
            password='rmivuxg',
            database='route_optimization'
        )
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT vehicle_type, capacity_kg, max_locations, cost_per_km, max_distance_km
            FROM vehicle_configs 
            WHERE hub_id = %s AND is_active = TRUE 
            ORDER BY vehicle_type
        """, (hub_id,))
        
        vehicles = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Convert to VEHICLES format
        vehicles_dict = {}
        for v in vehicles:
            vehicle_type = v['vehicle_type']
            vehicles_dict[vehicle_type] = {
                "capacity": float(v['capacity_kg']),
                "cost_per_km": float(v['cost_per_km']),
                "max_locations": int(v['max_locations']),
                "cost_type": "round_trip",  # Default to round_trip
                "range_km": float(v['max_distance_km']),
                "min_cost": 0,  # Default min_cost
                "hub_id": [hub_id]  # Only for this hub
            }
        
        print(f"Loaded {len(vehicles_dict)} vehicles from database for hub {hub_id}")
        return vehicles_dict
        
    except Exception as e:
        print(f"Error loading vehicles from database: {e}")
        print("Falling back to hardcoded VEHICLES")
        return None

# ------------------------------------------------
# Vehicle catalogue (fallback)
# ------------------------------------------------
VEHICLES = {
    "Tata Ace": {
        "capacity": 850, "cost_per_km": 13, "max_locations": 10,
        "cost_type": "round_trip", "range_km": 150, "min_cost": 1000,
        "hub_id": [1,2,3,8,13,18,29,31,32]
    },
    "Dost-1": {
        "capacity": 1251, "cost_per_km": 16, "max_locations": 10,
        "cost_type": "round_trip", "range_km": 500, "min_cost": 1200,
        "hub_id": [3,14,18,31,32]
    },
    "Dost-2": {
        "capacity": 1551, "cost_per_km": 16, "max_locations": 10,
        "cost_type": "round_trip", "range_km": 1000, "min_cost": 1200,
        "hub_id": [20,30]
    },
    "Bolero-1": {
        "capacity": 1551, "cost_per_km": 17, "max_locations": 10,
        "cost_type": "round_trip", "range_km": 1500, "min_cost": 1500,
        "hub_id": [2,3,8,10,13,18,19,28,29,31,32]
    },
    "Bolero-3": {
        "capacity": 1851, "cost_per_km": 17, "max_locations": 10,
        "cost_type": "round_trip", "range_km": 1000, "min_cost": 1500,
        "hub_id": [1,22,25,26]
    },
    "Tata 407": {
        "capacity": 2501, "cost_per_km": 20, "max_locations": 10,
        "cost_type": "round_trip", "range_km": 1500, "min_cost": 2000,
        "hub_id": [28,30,32]
    },
    "14ft": {
        "capacity": 3500, "cost_per_km": 23, "max_locations": 10,
        "cost_type": "round_trip", "range_km": 1000, "min_cost": 2500,
        "hub_id": [1,2,3,13,18,30,31]
    },
    "20ft_container": {
        "capacity": 8001, "cost_per_km": 35, "max_locations": 1,
        "cost_type": "one_way", "range_km": 5000, "min_cost": 5500,
        "hub_id": [8, 31, 26, 18, 13, 10, 20, 22, 19, 14, 22,23, 28,
                   1, 2, 3, 5, 16, 29, 25,30,32]
    },
    "32ft_container_sxl": {
        "capacity": 10001, "cost_per_km": 40, "max_locations": 1,
        "cost_type": "one_way", "range_km": 5000, "min_cost": 9000,
        "hub_id": [8, 31, 26, 18, 13, 10, 20, 22, 19, 14, 23, 28,
                   1, 2, 3, 5, 16, 29, 25,30,32]
    },
    "32ft_container_mxl": {
        "capacity": 18000, "cost_per_km": 60, "max_locations": 1,
        "cost_type": "one_way", "range_km": 5000, "min_cost": 12000,
        "hub_id": [8, 31, 26, 18, 13, 10, 20, 22, 19, 14, 23, 28,
                   1, 2, 3, 5, 16, 29,25, 30,32]
    },
    "25MT Tarus": {
        "capacity": 30000, "cost_per_km": 65, "max_locations": 1,
        "cost_type": "one_way", "range_km": 5000, "min_cost": 12000,
        "hub_id": [8, 31, 26, 18, 13, 10, 20, 22, 19, 14, 23, 28,
                   1, 2, 3, 5, 16, 29,25, 30,32]
    },
}

def exceeds_range(total_dist, vehicle_name):
    return total_dist > VEHICLES[vehicle_name]['range_km']

# ------------------------------------------------
# Hub-ID -> "STATE+PIN" map
# ------------------------------------------------
HUB_PINCODE_MAP = {
    8:  "TAMIL NADU+600060",
    30: "TAMIL NADU+641005",
    1:  "KARNATAKA+562114",
    2:  "HARYANA+131029",
    3:  "UTTAR PRADESH+226008",
    31:  "MAHARASHTRA+402107",
    26: "KARNATAKA+580024",
    5:  "TELANGANA+500070",
    18: "MAHARASHTRA+421506",
    28: "ANDHRA PRADESH+522001",
    13: "RAJASTHAN+302013",
    29: "PUNJAB+141003",
    10: "UTTAR PRADESH+243302",
    20: "JAMMU AND KASHMIR+192301",
    22: "MADHYA PRADESH+462039",
    19: "UTTAR PRADESH+273403",
    14: "WEST BENGAL+712310",
    23: "BIHAR+800023",
    16: "UTTAR PRADESH+221108",
    25: "ODISHA+754021",
    32: "TELANGANA+501510"
}

# ------------------------------------------------
# TSP / Reorder Logic
# ------------------------------------------------
def sequence_locations(trip, calculate_dist):
    """
    Optimize the sequence of locations in a trip to minimize total distance (Held-Karp TSP).
    """
    middle = trip[1:-1]
    if len(middle) <= 1:
        return trip

    n = len(middle)
    all_distances = {}
    for i, loc1 in enumerate(middle):
        for j, loc2 in enumerate(middle):
            if i != j:
                dist_val = calculate_dist(loc1[0], loc2[0])
                all_distances[(i, j)] = dist_val

    dp = {}
    parent = {}

    # Start from index 0
    for i in range(1, n):
        if (0, i) in all_distances:
            dp[(1 << i, i)] = all_distances[(0, i)]

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            subset_mask = sum(1 << s for s in subset)
            for next_loc in subset:
                prev_subset_mask = subset_mask & ~(1 << next_loc)
                dp_key = (subset_mask, next_loc)
                if dp_key not in dp:
                    dp[dp_key] = float('inf')
                min_val = float('inf')
                min_prev_loc = None
                for prev_loc in subset:
                    if prev_loc != next_loc \
                       and (prev_subset_mask, prev_loc) in dp \
                       and (prev_loc, next_loc) in all_distances:
                        cost_temp = dp[(prev_subset_mask, prev_loc)] + all_distances[(prev_loc, next_loc)]
                        if cost_temp < min_val:
                            min_val = cost_temp
                            min_prev_loc = prev_loc
                if min_val < float('inf'):
                    dp[dp_key] = min_val
                    parent[dp_key] = min_prev_loc

    final_mask = (1 << n) - 1
    last_node = min(
        (i for i in range(1, n) if (final_mask, i) in dp and (i, 0) in all_distances),
        key=lambda i: dp[(final_mask, i)] + all_distances[(i, 0)],
        default=None
    )
    if last_node is None:
        return trip

    best_path = [last_node]
    curr_node = last_node
    curr_mask = final_mask
    while curr_node is not None and curr_node != 0:
        curr_node = parent.get((curr_mask, curr_node))
        if curr_node is not None:
            best_path.append(curr_node)
            curr_mask &= ~(1 << curr_node)

    best_path.reverse()

    optimized_trip = [trip[0]]  # hub at start
    for idx in best_path:
        optimized_trip.append(middle[idx])
    optimized_trip.append(trip[-1])  # hub at end
    return optimized_trip

# ------------------------------------------------
# Z-SCORE HELPER FUNCTIONS
# ------------------------------------------------
def compute_all_pairwise_distances(df):
    """Returns a list of all distances between unique locations in df."""
    unique_locs = df['source_state_pincode'].unique()
    dist_list = []
    for i, loc1 in enumerate(unique_locs):
        for j in range(i+1, len(unique_locs)):
            loc2 = unique_locs[j]
            dist_val = calculate_dist(loc1, loc2)
            if dist_val is not None:
                dist_list.append(dist_val)
    return dist_list

def compute_neighbors_count(df, threshold_km):
    """
    Count how many other locations are within 'threshold_km'
    of each location's 'source_state_pincode'.
    """
    unique_locs = df['source_state_pincode'].unique()
    print(f"[NEIGHBORS] Computing neighbor counts for {len(unique_locs)} unique locations (threshold: {threshold_km}km)")
    
    loc_neighbors = {}
    total_pairs = len(unique_locs) * (len(unique_locs) - 1) // 2
    processed_pairs = 0
    
    for i, loc1 in enumerate(unique_locs):
        count = 0
        print(f"[NEIGHBORS] Processing location {i+1}/{len(unique_locs)}: {loc1}")
        
        for j, loc2 in enumerate(unique_locs):
            if loc1 == loc2:
                continue
            processed_pairs += 1
            if processed_pairs % 10 == 0:
                print(f"[NEIGHBORS] Processed {processed_pairs}/{total_pairs} location pairs...")
                
            dist_val = calculate_dist(loc1, loc2)
            if dist_val is not None and dist_val <= threshold_km:
                count += 1
                print(f"[NEIGHBORS] {loc1} -> {loc2}: {dist_val:.2f}km (within threshold)")
            elif dist_val is not None:
                print(f"[NEIGHBORS] {loc1} -> {loc2}: {dist_val:.2f}km (beyond threshold)")
        
        loc_neighbors[loc1] = count
        print(f"[NEIGHBORS] {loc1} has {count} neighbors within {threshold_km}km")

    df['neighbors_count'] = df['source_state_pincode'].map(loc_neighbors)
    print(f"[NEIGHBORS] Completed neighbor count computation")
    return df

def add_priority_zscore(df):
    """
    Create a combined priority score favoring:
      1) heavier load,
      2) more neighbors,
      3) closer to the hub (smaller distance_from_hub => higher score).
    """
    # Ensure columns exist
    for col in ['weight', 'neighbors_count', 'distance_from_hub']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    mean_w = df['weight'].mean();   std_w = df['weight'].std() or 1e-9
    mean_n = df['neighbors_count'].mean(); std_n = df['neighbors_count'].std() or 1e-9
    mean_d = df['distance_from_hub'].mean(); std_d = df['distance_from_hub'].std() or 1e-9

    df['z_weight'] = (df['weight'] - mean_w) / std_w
    df['z_neighbors'] = (df['neighbors_count'] - mean_n) / std_n
    df['z_dist_hub'] = (df['distance_from_hub'] - mean_d) / std_d

    # Invert distance => closer = higher score
    df['z_dist_hub_inverted'] = -df['z_dist_hub']

    # Combine them (tune weights alpha, beta, gamma as desired)
    alpha = 1.0
    beta = 1.0
    gamma = 1.0

    df['zscore_priority'] = (
        alpha * df['z_weight'] +
        beta  * df['z_neighbors'] +
        gamma * df['z_dist_hub_inverted']
    )

    return df

# ------------------------------------------------
# Cost calculation
# ------------------------------------------------
def calculate_trip_cost(trip):
    """
    Returns:
        total_distance (km), total_weight (kg), final_cost (INR), cost_per_kg (INR/kg)
    """
    if len(trip) < 2:
        return 0, 0, 0, 0

    total_distance = 0
    total_weight   = 0
    vtype = trip[1][2]

    if vtype not in VEHICLES:
        return 0, 0, 0, 0

    specs      = VEHICLES[vtype]
    cost_type  = specs["cost_type"]
    hub_loc    = trip[0][0]

    if cost_type == "one_way":
        last = trip[1][0]
        for loc, wt, _, _ in trip[1:-1]:
            total_weight += wt
            d = calculate_dist(last, loc)
            if d is not None:
                total_distance += d
            last = loc
        d_back = calculate_dist(last, hub_loc)
        if d_back is not None:
            total_distance += d_back

    elif cost_type == "round_trip":
        last = hub_loc
        for loc, wt, _, _ in trip[1:-1]:
            total_weight += wt
            d = calculate_dist(last, loc)
            if d is not None:
                total_distance += d
            last = loc
        d_back = calculate_dist(last, hub_loc)
        if d_back is not None:
            total_distance += d_back

    base_cost  = total_distance * specs["cost_per_km"]
    min_cost   = specs.get("min_cost", 0)
    final_cost = max(base_cost, min_cost)

    return (
        total_distance,
        total_weight,
        final_cost,
        (final_cost / total_weight) if total_weight else 0
    )

# ------------------------------------------------
# Vehicle utilities
# ------------------------------------------------
def sort_vehicles_by_capacity_desc(vehicle_type_filter):
    """
    Return the list of vehicle names (with matching cost_type)
    sorted by capacity in DESC order.
    """
    filtered = {
        v: specs for v, specs in VEHICLES.items()
        if specs['cost_type'] == vehicle_type_filter
    }
    sorted_list = sorted(filtered.keys(), key=lambda x: filtered[x]['capacity'], reverse=True)
    return sorted_list

# ------------------------------------------------
# TRIP BUILDER
# ------------------------------------------------
def build_trip_with_vehicle(location_weights, anchor_idx, vehicle_name):
    if anchor_idx not in location_weights.index:
        return None, []

    row = location_weights.loc[anchor_idx]
    hub = row['hub_state_pincode']
    hub_id = row['hub_id']
    anchor_loc = row['source_state_pincode']
    anchor_wt = row['weight']
    anchor_pnums = row['pickup_number']

    if hub_id not in VEHICLES[vehicle_name]["hub_id"]:
        return None, []

    cap = VEHICLES[vehicle_name]["capacity"]
    max_locs = VEHICLES[vehicle_name]["max_locations"]
    if anchor_wt > cap:
        return None, []

    current_trip_stops = [(anchor_loc, anchor_wt, vehicle_name, anchor_pnums)]
    used_indices = [anchor_idx]
    curr_wt = anchor_wt
    locs_in_trip = 1
    curr_loc = anchor_loc

    while True:
        if locs_in_trip >= max_locs:
            break
        remaining = location_weights.drop(index=used_indices, errors='ignore')
        if remaining.empty:
            break

        def d_lookup(r): return calculate_dist(curr_loc, r['source_state_pincode'])
        dists = remaining.apply(d_lookup, axis=1)
        nxt_idx = dists.idxmin()
        nxt_d = dists[nxt_idx]
        if pd.isna(nxt_d):
            break
        nxt_wt = remaining.loc[nxt_idx, 'weight']
        if curr_wt + nxt_wt <= cap:
            loc_name = remaining.loc[nxt_idx, 'source_state_pincode']
            pnums = remaining.loc[nxt_idx, 'pickup_number']
            current_trip_stops.append((loc_name, nxt_wt, vehicle_name, pnums))
            used_indices.append(nxt_idx)
            curr_loc = loc_name
            curr_wt += nxt_wt
            locs_in_trip += 1
        else:
            break

    candidate_trip = [(hub, 0, vehicle_name, [])] + current_trip_stops + [(hub, 0, vehicle_name, [])]

    # Range check
    td, _, _, _ = calculate_trip_cost(candidate_trip)
    if exceeds_range(td, vehicle_name):
        return None, []

    return candidate_trip, used_indices

def create_trips_iterative(location_weights, vehicle_type_filter):
    trips, alerts = [], []
    total_locations = len(location_weights)
    processed_locations = 0

    print(f"[TRIPS] Starting iterative trip creation for {total_locations} locations")

    # get vehicles sorted by capacity desc
    vehicle_order = sort_vehicles_by_capacity_desc(vehicle_type_filter)
    print(f"[TRIPS] Available vehicles: {vehicle_order}")

    while not location_weights.empty:
        remaining = len(location_weights)
        print(f"[TRIPS] Processing trip {len(trips) + 1}: {remaining} locations remaining")
        
        # 1) pick your anchor point
        anchor_idx = location_weights['zscore_priority'].idxmax()
        anchor_row = location_weights.loc[anchor_idx]
        print(f"[TRIPS] Selected anchor: {anchor_row['source_state_pincode']} (priority: {anchor_row['zscore_priority']:.3f})")

        best_trip = None
        best_cpk  = float('inf')
        best_idxs = []

        # 2) try every vehicle on that anchor, record its cost/kg
        for v in vehicle_order:
            print(f"[TRIPS] Trying vehicle: {v}")
            candidate_trip, used_idxs = build_trip_with_vehicle(
                location_weights, anchor_idx, v
            )
            if not candidate_trip:
                print(f"[TRIPS] No valid trip for vehicle {v}")
                continue

            _, _, _, cpk = calculate_trip_cost(candidate_trip)
            print(f"[TRIPS] Vehicle {v}: {len(candidate_trip)} stops, cost/kg: Rs{cpk:.2f}")
            # track the lowest cost/kg
            if cpk < best_cpk:
                best_cpk  = cpk
                best_trip = candidate_trip
                best_idxs = used_idxs

        # 3) if we found at least one valid trip
        if best_trip:
            print(f"[TRIPS] Best trip found: {len(best_trip)} stops, cost/kg: Rs{best_cpk:.2f}")
            # TSP-optimize the stops if more than one pickup
            if len(best_trip) > 3:
                print(f"[TRIPS] Optimizing route sequence for {len(best_trip)} stops...")
                best_trip = sequence_locations(best_trip, calculate_dist)

            trips.append(best_trip)
            processed_locations += len(best_idxs)
            print(f"[TRIPS] Trip {len(trips)} completed: {processed_locations}/{total_locations} locations processed")
            # remove those served points from the pool
            location_weights.drop(index=best_idxs, inplace=True, errors='ignore')

        else:
            print(f"[TRIPS] No suitable vehicle found for {anchor_row['source_state_pincode']}")
            alerts.append(
                f"No suitable round-trip vehicle for {anchor_row['source_state_pincode']}"
            )
            location_weights.drop(index=anchor_idx, inplace=True, errors='ignore')
            processed_locations += 1

    print(f"[TRIPS] Completed: {len(trips)} trips created, {len(alerts)} alerts")
    return trips, alerts

# ------------------------------------------------
# LARGE LOAD HANDLING
# ------------------------------------------------
def select_one_way_vehicle_for_large(load, hub_id):
    """
    Return the one-way vehicle whose capacity >= load and
    is as close to 'load' as possible (the smallest capacity that still fits).
    """
    candidates = []
    for v_name, v_specs in VEHICLES.items():
        if v_specs['cost_type'] == 'one_way' and hub_id in v_specs['hub_id'] and v_specs['capacity'] >= load:
            candidates.append((v_name, v_specs['capacity']))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]

def create_single_trips_for_large(df_large):
    trips, alerts = [], []
    for _, row in df_large.iterrows():
        hub_loc = row['hub_state_pincode']
        hub_id = row['hub_id']
        load = row['weight']
        loc = row['source_state_pincode']
        pnums = row['pickup_number']

        v_name = select_one_way_vehicle_for_large(load, hub_id)
        if not v_name:
            alerts.append(f"No suitable one-way vehicle for {loc} (load {load})")
            trip = [(hub_loc, 0, "Special", []),
                    (loc, load, "Special", pnums),
                    (hub_loc, 0, "Special", [])]
        else:
            trip = [(hub_loc, 0, v_name, []),
                    (loc, load, v_name, pnums),
                    (hub_loc, 0, v_name, [])]

        td, _, _, _ = calculate_trip_cost(trip)
        if exceeds_range(td, trip[1][2]):
            alerts.append(
                f"Trip for {loc} exceeds {v_name} range "
                f"({td:.1f} km > {VEHICLES[v_name]['range_km']} km)"
            )
            continue

        trips.append(trip)
    return trips, alerts

# ------------------------------------------------
# CSV INPUT/OUTPUT FUNCTIONS
# ------------------------------------------------
def read_csv_data(csv_file_path):
    """
    Read pickup data from CSV file and prepare it for routing optimization.
    
    Expected CSV columns:
    - pickup_number: Unique identifier for pickup
    - material_quantity: Weight in kg
    - city_name: City name
    - state_name: State name  
    - location_pincode: Pincode
    - hub_id: Hub identifier
    """
    print(f"Reading CSV data from: {csv_file_path}")
    
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read {len(df)} records from CSV")
        
        # Validate required columns
        required_columns = ['pickup_number', 'material_quantity', 'city_name', 
                          'state_name', 'location_pincode', 'hub_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean and format data
        df = df[required_columns].copy()
        df['city_name'] = df['city_name'].str.upper()
        df['state_name'] = df['state_name'].str.upper()
        
        # Create location identifiers
        df['source_state_pincode'] = (
            df['state_name'] + '+' + df['location_pincode'].astype(str)
        )
        df['hub_state_pincode'] = df['hub_id'].map(HUB_PINCODE_MAP)
        
        # Convert material_quantity to numeric
        df['material_quantity'] = pd.to_numeric(df['material_quantity'], errors='coerce')
        df.dropna(subset=['material_quantity'], inplace=True)
        
        print(f"Processed {len(df)} valid records")
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

def calculate_distances_for_dataframe(df):
    """
    Calculate distances from hub to each pickup location.
    Remove locations where distance calculation fails.
    """
    print("Calculating distances from hub to pickup locations...")
    
    distances = []
    indices_to_remove = []
    
    for idx, row in df.iterrows():
        src = row['source_state_pincode']
        dst = row['hub_state_pincode']
        d = calculate_dist(src, dst)
        
        if d is None:
            print(f"Could not calculate distance for {src} -> {dst}")
            indices_to_remove.append(idx)
        else:
            distances.append(d)
    
    # Remove failed distance calculations
    df.drop(indices_to_remove, inplace=True)
    df['distance_from_hub'] = distances
    df.reset_index(drop=True, inplace=True)
    
    print(f"Calculated distances for {len(df)} locations")
    return df

def write_routes_to_csv(trips, alerts, hub_id, output_file_path, selected_model=None, selection_reason=None):
    """
    Write optimized routes to CSV file with better formatting.
    
    CSV format:
    - trip_number: Sequential trip number
    - vehicle_type: Type of vehicle used
    - stop_sequence: Order of stops in trip
    - location: Pickup location (STATE+PINCODE format)
    - pickup_numbers: Comma-separated pickup numbers
    - weight_kg: Weight at this stop
    - total_distance_km: Total trip distance (only on first row of each trip)
    - total_weight_kg: Total trip weight (only on first row of each trip)
    - cost_inr: Total trip cost (only on first row of each trip)
    - cost_per_kg: Cost per kg (only on first row of each trip)
    - algorithm_used: Algorithm that generated this route (if available)
    """
    print(f"Writing optimized routes to: {output_file_path}")
    
    # Prepare data for CSV
    csv_data = []
    
    for trip_num, trip in enumerate(trips, 1):
        # Calculate trip metrics
        total_distance, total_weight, total_cost, cost_per_kg = calculate_trip_cost(trip)
        
        # Process each stop in the trip
        for stop_seq, stop in enumerate(trip[1:-1], 1):  # Skip hub at start/end
            location, weight, vehicle_type, pickup_numbers = stop
            
            # Only show trip totals on the first stop of each trip
            show_trip_totals = (stop_seq == 1)
            
            csv_data.append({
                'trip_number': trip_num,
                'vehicle_type': vehicle_type,
                'stop_sequence': stop_seq,
                'location': location,
                'pickup_numbers': ','.join(pickup_numbers) if pickup_numbers else '',
                'weight_kg': weight,
                'total_distance_km': total_distance if show_trip_totals else '',
                'total_weight_kg': total_weight if show_trip_totals else '',
                'cost_inr': total_cost if show_trip_totals else '',
                'cost_per_kg': cost_per_kg if show_trip_totals else '',
                'algorithm_used': selected_model or 'Unknown' if show_trip_totals else ''
            })
    
    # Create DataFrame and write to CSV
    routes_df = pd.DataFrame(csv_data)
    routes_df.to_csv(output_file_path, index=False)
    
    print(f"Successfully wrote {len(routes_df)} route stops to CSV")
    print(f"Generated {len(trips)} optimized trips")
    
    # Print summary
    total_distance = sum(calculate_trip_cost(trip)[0] for trip in trips)
    total_cost = sum(calculate_trip_cost(trip)[2] for trip in trips)
    total_weight = sum(calculate_trip_cost(trip)[1] for trip in trips)
    
    print(f"\nRoute Optimization Summary:")
    print(f"   Hub ID: {hub_id}")
    print(f"   Algorithm Used: {selected_model or 'Unknown'}")
    if selection_reason:
        print(f"   Selection Reason: {selection_reason}")
    print(f"   Total Trips: {len(trips)}")
    print(f"   Total Distance: {total_distance:.2f} km")
    print(f"   Total Weight: {total_weight:.2f} kg")
    print(f"   Total Cost: Rs.{total_cost:.2f}")
    print(f"   Average Cost per kg: Rs.{total_cost/total_weight:.2f}" if total_weight > 0 else "   Average Cost per kg: Rs.0.00")
    
    if alerts:
        print(f"\nAlerts ({len(alerts)}):")
        for alert in alerts:
            print(f"   - {alert}")

def create_sample_csv(sample_file_path):
    """
    Create a sample CSV file with the required format and sample data.
    """
    print(f"Creating sample CSV file: {sample_file_path}")
    
    sample_data = {
        'pickup_number': [
            'P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010',
            'P011', 'P012', 'P013', 'P014', 'P015', 'P016', 'P017', 'P018', 'P019', 'P020'
        ],
        'material_quantity': [
            150, 200, 300, 120, 450, 180, 250, 350, 100, 400,
            280, 160, 220, 380, 140, 320, 190, 270, 310, 170
        ],
        'city_name': [
            'Chennai', 'Chennai', 'Chennai', 'Chennai', 'Chennai',
            'Bangalore', 'Bangalore', 'Bangalore', 'Bangalore', 'Bangalore',
            'Mumbai', 'Mumbai', 'Mumbai', 'Mumbai', 'Mumbai',
            'Delhi', 'Delhi', 'Delhi', 'Delhi', 'Delhi'
        ],
        'state_name': [
            'Tamil Nadu', 'Tamil Nadu', 'Tamil Nadu', 'Tamil Nadu', 'Tamil Nadu',
            'Karnataka', 'Karnataka', 'Karnataka', 'Karnataka', 'Karnataka',
            'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra',
            'Delhi', 'Delhi', 'Delhi', 'Delhi', 'Delhi'
        ],
        'location_pincode': [
            600001, 600002, 600003, 600004, 600005,
            560001, 560002, 560003, 560004, 560005,
            400001, 400002, 400003, 400004, 400005,
            110001, 110002, 110003, 110004, 110005
        ],
        'hub_id': [8] * 20  # All using hub 8 (Chennai)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv(sample_file_path, index=False)
    
    print(f"Created sample CSV with {len(sample_df)} records")
    print(f"Sample data includes pickups from Chennai, Bangalore, Mumbai, and Delhi")
    print(f"All assigned to Hub 8 (Chennai)")

# ------------------------------------------------
# MODEL COMPARISON FUNCTIONS (from model_comparison_worker.py)
# ------------------------------------------------
def calculate_geographic_density(location_weights_df):
    """Calculate geographic density to determine HDBSCAN suitability."""
    if len(location_weights_df) < 10:
        return 0.0

    # Get unique states
    states = location_weights_df['source_state_pincode'].str.split('+').str[0].unique()
    state_count = len(states)
    location_count = len(location_weights_df)

    # Calculate state concentration (higher = more concentrated)
    state_concentration = location_count / state_count if state_count > 0 else 0

    # Calculate intra-state clustering potential
    same_state_pairs = 0
    total_pairs = 0

    for _, row1 in location_weights_df.iterrows():
        for _, row2 in location_weights_df.iterrows():
            if row1.name != row2.name:
                total_pairs += 1
                state1 = row1['source_state_pincode'].split('+')[0]
                state2 = row2['source_state_pincode'].split('+')[0]
                if state1 == state2:
                    same_state_pairs += 1

    intra_state_ratio = same_state_pairs / total_pairs if total_pairs > 0 else 0

    # Combined density score (0-1, higher = better for HDBSCAN)
    density_score = min(1.0, (state_concentration / 20) * 0.6 + intra_state_ratio * 0.4)

    return density_score

def run_hdbscan_model(location_weights_small, df_large_for_trips, hub_id):
    """Run the HDBSCAN clustering based routing model."""
    print(f"\n[ROCKET] Running HDBSCAN Model for Hub {hub_id}")
    start_time = time.time()

    if not HDBSCAN_AVAILABLE or not HDBSCAN_ROUTER_AVAILABLE:
        print("[WARNING] HDBSCAN not available, using Z-score fallback")
        return run_zscore_model(location_weights_small, df_large_for_trips, hub_id)

    # Intelligent HDBSCAN suitability check
    location_count = len(location_weights_small)
    if location_count < 50:
        print(f"[WARNING] Dataset too small for effective HDBSCAN clustering ({location_count} locations)")
        print("   HDBSCAN works best with >50 locations. Using Z-score for better performance.")
        return run_zscore_model(location_weights_small, df_large_for_trips, hub_id)

    # Check geographic density
    geographic_density = calculate_geographic_density(location_weights_small)
    if geographic_density < 0.3:
        print(f"[WARNING] Low geographic density ({geographic_density:.2f}) - locations too dispersed")
        print("   Z-Score better suited for sparse geographic distributions.")
        return run_zscore_model(location_weights_small, df_large_for_trips, hub_id)

    # Process small loads with HDBSCAN
    if not location_weights_small.empty and len(location_weights_small) >= 8:
        # Initialize HDBSCAN router
        router = HDBSCANEnhancedRouter(
            min_cluster_size=max(8, len(location_weights_small) // 10),
            min_samples=3,
            cluster_selection_epsilon=0.1
        )

        # Add distance from hub for HDBSCAN features
        if 'distance_from_hub' not in location_weights_small.columns:
            distances = []
            for _, row in location_weights_small.iterrows():
                d = calculate_dist(row['source_state_pincode'], row['hub_state_pincode'])
                distances.append(d if d is not None else 0)
            location_weights_small['distance_from_hub'] = distances

        # Run enhanced routing
        try:
            trips_small, alerts_small, performance_metrics = router.enhanced_route_optimization(
                location_weights_small, hub_id
            )
        except Exception as e:
            print(f"[WARNING] HDBSCAN failed: {e}, falling back to Z-score")
            location_weights_small = compute_neighbors_count(location_weights_small, 60)
            location_weights_small = add_priority_zscore(location_weights_small)
            trips_small, alerts_small = create_trips_iterative(
                location_weights_small, vehicle_type_filter='round_trip'
            )
    else:
        # Fallback to Z-score for small datasets
        if not location_weights_small.empty:
            location_weights_small = compute_neighbors_count(location_weights_small, 60)
            location_weights_small = add_priority_zscore(location_weights_small)
            trips_small, alerts_small = create_trips_iterative(
                location_weights_small, vehicle_type_filter='round_trip'
            )
        else:
            trips_small, alerts_small = [], []

    # Process large loads (same as Z-score)
    if not df_large_for_trips.empty:
        df_large_for_trips = compute_neighbors_count(df_large_for_trips, 60)
        df_large_for_trips = add_priority_zscore(df_large_for_trips)
        trips_large, alerts_large = create_single_trips_for_large(df_large_for_trips)
    else:
        trips_large, alerts_large = [], []

    processing_time = time.time() - start_time

    # Combine results
    all_trips = trips_small + trips_large
    all_alerts = alerts_small + alerts_large

    # Calculate metrics
    total_cost = sum(calculate_trip_cost(trip)[2] for trip in all_trips)
    total_distance = sum(calculate_trip_cost(trip)[0] for trip in all_trips)
    total_weight = sum(calculate_trip_cost(trip)[1] for trip in all_trips)

    results = {
        'model_name': 'HDBSCAN',
        'trips': all_trips,
        'alerts': all_alerts,
        'total_cost': total_cost,
        'total_distance': total_distance,
        'total_weight': total_weight,
        'trip_count': len(all_trips),
        'cost_per_kg': total_cost / total_weight if total_weight > 0 else 0,
        'processing_time': processing_time,
        'avg_distance_per_trip': total_distance / len(all_trips) if all_trips else 0
    }

    print(f"[SUCCESS] HDBSCAN Model completed in {processing_time:.2f}s")
    print(f"   Trips: {len(all_trips)}, Cost: Rs{total_cost:.0f}, Distance: {total_distance:.1f}km")

    return results

def run_zscore_model(location_weights_small, df_large_for_trips, hub_id):
    """Run the traditional Z-score based routing model."""
    print(f"\n[TARGET] Running Z-Score Model for Hub {hub_id}")
    start_time = time.time()

    # Process small loads
    if not location_weights_small.empty:
        print(f"[Z-SCORE] Processing {len(location_weights_small)} small load locations")
        # Calculate neighbors and Z-scores
        print(f"[Z-SCORE] Computing neighbor counts...")
        location_weights_small = compute_neighbors_count(location_weights_small, 60)
        print(f"[Z-SCORE] Computing Z-score priorities...")
        location_weights_small = add_priority_zscore(location_weights_small)

        # Create trips using iterative approach
        print(f"[Z-SCORE] Creating trips iteratively...")
        trips_small, alerts_small = create_trips_iterative(
            location_weights_small, vehicle_type_filter='round_trip'
        )
    else:
        trips_small, alerts_small = [], []

    # Process large loads
    if not df_large_for_trips.empty:
        df_large_for_trips = compute_neighbors_count(df_large_for_trips, 60)
        df_large_for_trips = add_priority_zscore(df_large_for_trips)
        trips_large, alerts_large = create_single_trips_for_large(df_large_for_trips)
    else:
        trips_large, alerts_large = [], []

    processing_time = time.time() - start_time

    # Combine results
    all_trips = trips_small + trips_large
    all_alerts = alerts_small + alerts_large

    # Calculate metrics
    total_cost = sum(calculate_trip_cost(trip)[2] for trip in all_trips)
    total_distance = sum(calculate_trip_cost(trip)[0] for trip in all_trips)
    total_weight = sum(calculate_trip_cost(trip)[1] for trip in all_trips)

    results = {
        'model_name': 'Z-Score',
        'trips': all_trips,
        'alerts': all_alerts,
        'total_cost': total_cost,
        'total_distance': total_distance,
        'total_weight': total_weight,
        'trip_count': len(all_trips),
        'cost_per_kg': total_cost / total_weight if total_weight > 0 else 0,
        'processing_time': processing_time,
        'avg_distance_per_trip': total_distance / len(all_trips) if all_trips else 0
    }

    print(f"[SUCCESS] Z-Score Model completed in {processing_time:.2f}s")
    print(f"   Trips: {len(all_trips)}, Cost: Rs{total_cost:.0f}, Distance: {total_distance:.1f}km")

    return results

def select_best_model_by_cost_per_kg(hdbscan_results, zscore_results, hub_id):
    """
    Automatically select the best model based on cost per kg.
    """
    print(f"\n[TARGET] AUTOMATIC MODEL SELECTION FOR HUB {hub_id}")
    print("=" * 60)

    hdbscan_cost_per_kg = hdbscan_results['cost_per_kg']
    zscore_cost_per_kg = zscore_results['cost_per_kg']
    hdbscan_total_cost = hdbscan_results['total_cost']
    zscore_total_cost = zscore_results['total_cost']

    print(f"[CHART] Cost Comparison:")
    print(f"   HDBSCAN: Rs{hdbscan_cost_per_kg:.2f}/kg (Total: Rs{hdbscan_total_cost:,.0f})")
    print(f"   Z-Score: Rs{zscore_cost_per_kg:.2f}/kg (Total: Rs{zscore_total_cost:,.0f})")

    # Calculate percentage difference
    cost_difference = abs(hdbscan_cost_per_kg - zscore_cost_per_kg)
    min_cost = min(hdbscan_cost_per_kg, zscore_cost_per_kg)
    cost_difference_percent = (cost_difference / min_cost * 100) if min_cost > 0 else 0

    print(f"   Difference: {cost_difference_percent:.2f}%")

    # Selection logic
    if cost_difference_percent < 1.0:
        # Less than 1% difference - choose Z-Score for simplicity and reliability
        selected_model = "Z-Score"
        selection_reason = f"Minimal cost difference ({cost_difference_percent:.2f}%) - Z-Score selected for simplicity and reliability"
    elif hdbscan_cost_per_kg < zscore_cost_per_kg:
        # HDBSCAN is better
        savings_percent = ((zscore_cost_per_kg - hdbscan_cost_per_kg) / zscore_cost_per_kg * 100)
        selected_model = "HDBSCAN"
        selection_reason = f"HDBSCAN offers {savings_percent:.1f}% cost savings (Rs{zscore_cost_per_kg - hdbscan_cost_per_kg:.2f}/kg less)"
    else:
        # Z-Score is better
        savings_percent = ((hdbscan_cost_per_kg - zscore_cost_per_kg) / hdbscan_cost_per_kg * 100)
        selected_model = "Z-Score"
        selection_reason = f"Z-Score offers {savings_percent:.1f}% cost savings (Rs{hdbscan_cost_per_kg - zscore_cost_per_kg:.2f}/kg less)"

    print(f"\n[TROPHY] SELECTED: {selected_model}")
    print(f"[IDEA] REASON: {selection_reason}")

    return selected_model, selection_reason

# ------------------------------------------------
# MAIN ROUTING FUNCTION
# ------------------------------------------------
def optimize_routes_from_csv(input_csv_path, output_csv_path, hub_id, job_id=None):
    """
    Main function to optimize routes from CSV input and output to CSV.
    Uses HDBSCAN clustering algorithm for spatial route optimization.
    """
    print(f"Starting CSV-based Route Optimization with HDBSCAN Clustering")
    print(f"   Input: {input_csv_path}")
    print(f"   Output: {output_csv_path}")
    print(f"   Hub ID: {hub_id}")
    print("=" * 60)
    
    # Update status: Starting optimization
    if job_id:
        update_job_status_message(job_id, "Starting route optimization...")
    
    # Load vehicles from database for this hub
    global VEHICLES
    db_vehicles = load_vehicles_from_db(hub_id)
    if db_vehicles:
        VEHICLES = db_vehicles
        print(f"Using {len(VEHICLES)} vehicles from database for hub {hub_id}")
    else:
        print(f"Using hardcoded VEHICLES (fallback)")
    
    try:
        # Step 1: Read and prepare data
        if job_id:
            update_job_status_message(job_id, "Reading CSV data and validating locations...")
        df = read_csv_data(input_csv_path)
        
        if df.empty:
            raise ValueError("No valid data found in CSV file")
        
        print(f"[DATA] Loaded {len(df)} pickup records")
        
        # Step 2: Calculate distances
        if job_id:
            update_job_status_message(job_id, f"Calculating distances from hub to {len(df)} pickup locations...")
        df = calculate_distances_for_dataframe(df)
        
        if df.empty:
            raise ValueError("No valid locations remaining after distance calculation")
        
        # Step 3: Split into small and large loads
        if job_id:
            update_job_status_message(job_id, f"Analyzing load distribution: {len(df)} locations, preparing for optimization...")
        df_small = df[df['material_quantity'] < 3501].copy()
        df_large = df[df['material_quantity'] >= 3501].copy()
        
        print(f"Load Distribution:")
        print(f"   Small loads (< 3501 kg): {len(df_small)} locations")
        print(f"   Large loads (>= 3501 kg): {len(df_large)} locations")
        
        # Step 4: Prepare small loads (group by location)
        location_weights_small = None
        if not df_small.empty:
            location_weights_small = (
                df_small
                .groupby(['source_state_pincode', 'hub_state_pincode', 'hub_id'],
                         as_index=False)
                .agg({
                    'material_quantity': 'sum',
                    'pickup_number': lambda x: list(x)
                })
                .rename(columns={'material_quantity': 'weight'})
            )
        
        # Step 5: Prepare large loads (1-to-1)
        df_large_for_trips = None
        if not df_large.empty:
            df_large_for_trips = df_large[[
                'source_state_pincode', 'hub_state_pincode', 'hub_id',
                'material_quantity', 'pickup_number'
            ]].copy()
            df_large_for_trips.rename(columns={'material_quantity': 'weight'}, inplace=True)
            df_large_for_trips['pickup_number'] = df_large_for_trips[
                'pickup_number'].apply(lambda x: [x] if isinstance(x, str) else (
                    x if isinstance(x, list) else []))
        
        # Step 6: Run HDBSCAN algorithm only
        if job_id:
            update_job_status_message(job_id, "Running HDBSCAN clustering algorithm for route optimization...")
        print(f"\n[ROCKET] Running HDBSCAN Algorithm for Route Optimization...")
        hdbscan_results = run_hdbscan_model(
            location_weights_small.copy() if location_weights_small is not None and not location_weights_small.empty else pd.DataFrame(),
            df_large_for_trips.copy() if df_large_for_trips is not None and not df_large_for_trips.empty else pd.DataFrame(),
            hub_id
        )
        
        # Use HDBSCAN results directly (no comparison)
        best_results = hdbscan_results
        selected_model = hdbscan_results['model_name']
        selection_reason = "Using HDBSCAN clustering algorithm for spatial route optimization"
        
        if job_id:
            update_job_status_message(job_id, f"Finalizing routes and generating output file...")
        print(f"\n[UPLOAD] Finalizing Optimized Routes...")
        
        # Step 7: Write to CSV
        write_routes_to_csv(best_results['trips'], best_results['alerts'], hub_id, output_csv_path, selected_model, selection_reason)
        
        # Update job status to COMPLETE
        if job_id:
            update_job_status(job_id, 'COMPLETE', output_csv_path)
        
        # Print final summary
        print(f"\n[SUCCESS] Route optimization completed successfully!")
        print(f"[ROCKET] Algorithm Used: {selected_model}")
        print(f"[IDEA] Optimization Method: {selection_reason}")
        print(f"[PACKAGE] Total Weight: {best_results['total_weight']:,.0f} kg")
        print(f"[MONEY] Cost: Rs{best_results['total_cost']:,.0f} (Rs{best_results['cost_per_kg']:.2f}/kg)")
        print(f"[TRUCK] Trips: {best_results['trip_count']}")
        print(f"[DOCUMENT] Results written to: {output_csv_path}")
        
        return True
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error during route optimization: {e}\n{tb}")
        if job_id:
            update_job_status(job_id, 'ERROR', error_message=tb)
        return False

# ------------------------------------------------
# COMMAND LINE INTERFACE
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='CSV-based Route Optimization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize routes from input CSV
  python csv_route_optimizer.py input.csv output.csv 8
  
  # Create sample CSV file
  python csv_route_optimizer.py --create-sample sample_data.csv
  
Required CSV columns:
  pickup_number, material_quantity, city_name, state_name, location_pincode, hub_id
        """
    )
    
    parser.add_argument('input_file', nargs='?', help='Input CSV file path')
    parser.add_argument('output_file', nargs='?', help='Output CSV file path')
    parser.add_argument('hub_id', nargs='?', type=int, help='Hub ID for optimization')
    parser.add_argument('--job-id', help='Job ID for status updates (optional)')
    parser.add_argument('--create-sample', metavar='FILE', help='Create sample CSV file')
    parser.add_argument('--init-db', action='store_true', help='Initialize database and tables')
    
    args = parser.parse_args()
    
    # Handle database initialization
    if args.init_db:
        print("Initializing database and tables...")
        if create_database_and_tables():
            print("Database initialization completed successfully!")
        else:
            print("Database initialization failed!")
            sys.exit(1)
        return
    
    # Handle sample CSV creation
    if args.create_sample:
        create_sample_csv(args.create_sample)
        return
    
    # Validate required arguments
    if not all([args.input_file, args.output_file, args.hub_id]):
        parser.print_help()
        print(f"\nError: Missing required arguments")
        print(f"   Required: input_file, output_file, hub_id")
        print(f"   Use --create-sample to generate a sample CSV file")
        print(f"   Use --init-db to initialize database and tables")
        return
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # Initialize database if needed
    print("Checking database connection...")
    try:
        conn = get_mysql_connection()
        conn.close()
        print("Database connection successful")
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Please run: python csv_route_optimizer.py --init-db")
        return
    
    # Run optimization
    success = optimize_routes_from_csv(args.input_file, args.output_file, args.hub_id, args.job_id)
    
    if success:
        print(f"\nRoute optimization completed!")
        print(f"   Check the output file: {args.output_file}")
    else:
        print(f"\nRoute optimization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()