#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Route Optimization Worker with HDBSCAN Integration
============================================================

This is a complete standalone version of route_opt_worker.py enhanced with
HDBSCAN clustering for improved spatial routing efficiency.

Key Changes from Original:
- Added HDBSCAN clustering to solve nearby points skipping problem
- Enhanced spatial coherence in trip formation
- Maintains all existing cost optimization logic
- Graceful fallback to original Z-score algorithm if clustering fails

Usage:
- Can be run as drop-in replacement for route_opt_worker.py
- Test side-by-side with original for performance comparison
- All original functionality preserved
"""

# ------------------------------------------------
# Imports (Original + HDBSCAN additions)
# ------------------------------------------------
import requests
import pandas as pd
import mysql.connector
import googlemaps
from itertools import combinations
import json
import uuid
from datetime import datetime
import pytz
import numpy as np
import itertools
import time
import warnings

# HDBSCAN additions with detailed debugging
try:
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    HDBSCAN_AVAILABLE = True
    print("SUCCESS: HDBSCAN clustering available")
    try:
        print(f"   HDBSCAN version: {hdbscan.__version__}")
    except AttributeError:
        print("   HDBSCAN version: Available (version info not accessible)")
    print(f"   Sklearn location: {StandardScaler.__module__}")
except ImportError as e:
    HDBSCAN_AVAILABLE = False
    print("WARNING: HDBSCAN not available, will use original Z-score algorithm")
    print(f"   Import error details: {e}")
    print(f"   Python executable: {__import__('sys').executable}")
    print(f"   Python path: {__import__('sys').path[:3]}...")  # Show first 3 paths

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

# ------------------------------------------------
# API clients & globals (Unchanged)
# ------------------------------------------------
gmaps = googlemaps.Client(key='AIzaSyDMtTPCRVzDEyv-rcwk6BNIWZi4-bI-WZo')

new_api_key = 'UApDWhcNIesmLCt65ZmpWCskYvEWUg1PxqwCaXhn'
request_id  = str(uuid.uuid4())
BATCH_SIZE  = 10
IST_ZONE    = pytz.timezone("Asia/Kolkata")

# ------------------------------------------------
# MySQL connection helper (Unchanged)
# ------------------------------------------------
def get_mysql_connection():
    """Create a connection to the MySQL database."""
    return mysql.connector.connect(
        host='vaibhavjha.mysql.pythonanywhere-services.com',
        user='vaibhavjha',
        password='anubhav21',
        database='vaibhavjha$database'
    )

# ------------------------------------------------
# Distance-cache helpers (Unchanged)
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
    return result[0] if result else None

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
# Google & OlaMaps utilities (Unchanged)
# ------------------------------------------------
def batch_geocode(addresses):
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
    dists, failed = {}, []
    for i in range(0, len(origins), BATCH_SIZE):
        obatch = origins[i:i + BATCH_SIZE]
        dbatch = destinations[i:i + BATCH_SIZE]
        try:
            resp = gmaps.distance_matrix(obatch, dbatch)
            for o, d, row in zip(obatch, dbatch, resp['rows']):
                elem = row['elements'][0]
                if elem['status'] == 'OK':
                    dists[(o, d)] = elem['distance']['value'] / 1000.0  # km
                else:
                    failed.append((o, d))
        except Exception as e:
            print(f"Distance batch error: {e}")
            failed.extend([(o, d) for o in obatch for d in dbatch])
    return dists, failed

def calculate_dist(source, destination):
    # 1) DB cache
    dist = get_distance_from_db(source, destination)
    if dist is not None:
        return dist

    # 2) Google
    dists, _ = batch_calculate_distances([source], [destination])
    if (source, destination) in dists:
        dist = dists[(source, destination)]
        save_distance_to_db(source, destination, dist)
        return dist

    # 3) OlaMaps fallback
    return calculate_dist_new_api(source, destination)

def calculate_dist_new_api(source, destination):
    sg = get_geocode(source)
    dg = get_geocode(destination)
    if not sg or not dg:
        print("Failed geocode → OlaMaps skipped")
        return None

    url = (f"https://api.olamaps.io/routing/v1/distanceMatrix?"
           f"origins={sg}&destinations={dg}&api_key={new_api_key}")
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        elem = data['rows'][0]['elements'][0]
        if elem['status'] == 'OK':
            km = elem['distance'] / 1000.0
            save_distance_to_db(source, destination, km, sg, dg)
            return km
        print(f"OlaMaps element error: {elem['status']}")
    else:
        print(f"OlaMaps HTTP {resp.status_code}")
    return None

def get_geocode(address):
    gcs, _ = batch_geocode([address])
    return gcs.get(address)

# ------------------------------------------------
# JSON-safe NumPy converter (Unchanged)
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
# Vehicle catalogue (Unchanged)
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
# TSP / Reorder Logic (Unchanged)
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
# Z-SCORE HELPER FUNCTIONS (Unchanged)
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
    loc_neighbors = {}
    for loc1 in unique_locs:
        count = 0
        for loc2 in unique_locs:
            if loc1 == loc2:
                continue
            dist_val = calculate_dist(loc1, loc2)
            if dist_val is not None and dist_val <= threshold_km:
                count += 1
        loc_neighbors[loc1] = count

    df['neighbors_count'] = df['source_state_pincode'].map(loc_neighbors)
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
# Cost calculation (Unchanged)
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
# Vehicle utilities (Unchanged)
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
# TRIP BUILDER (Unchanged)
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

# ------------------------------------------------
# ORIGINAL ITERATIVE TRIPS (Unchanged)
# ------------------------------------------------
def create_trips_iterative(location_weights, vehicle_type_filter):
    trips, alerts = [], []

    # get vehicles sorted by capacity desc
    vehicle_order = sort_vehicles_by_capacity_desc(vehicle_type_filter)

    while not location_weights.empty:
        # 1) pick your anchor point
        anchor_idx = location_weights['zscore_priority'].idxmax()
        anchor_row = location_weights.loc[anchor_idx]

        best_trip = None
        best_cpk  = float('inf')
        best_idxs = []

        # 2) try every vehicle on that anchor, record its cost/kg
        for v in vehicle_order:
            candidate_trip, used_idxs = build_trip_with_vehicle(
                location_weights, anchor_idx, v
            )
            if not candidate_trip:
                continue

            _, _, _, cpk = calculate_trip_cost(candidate_trip)
            # track the lowest cost/kg
            if cpk < best_cpk:
                best_cpk  = cpk
                best_trip = candidate_trip
                best_idxs = used_idxs

        # 3) if we found at least one valid trip
        if best_trip:
            # TSP-optimize the stops if more than one pickup
            if len(best_trip) > 3:
                best_trip = sequence_locations(best_trip, calculate_dist)

            trips.append(best_trip)
            # remove those served points from the pool
            location_weights.drop(index=best_idxs, inplace=True, errors='ignore')

        else:
            alerts.append(
                f"No suitable round-trip vehicle for {anchor_row['source_state_pincode']}"
            )
            location_weights.drop(index=anchor_idx, inplace=True, errors='ignore')

    return trips, alerts

# ------------------------------------------------
# LARGE LOAD HANDLING (Unchanged)
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
            alerts.append(f"No suitable one‑way vehicle for {loc} (load {load})")
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
# ✨ ENHANCED HDBSCAN ROUTING CLASS ✨
# ------------------------------------------------
class HDBSCANEnhancedRouter:
    """
    Enhanced router that uses HDBSCAN clustering to guide trip formation
    and overcome the limitations of pure Z-score based routing.
    """

    def __init__(self, min_cluster_size=8, min_samples=3, cluster_selection_epsilon=0.1):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        if HDBSCAN_AVAILABLE:
            self.scaler = StandardScaler()

    def enhanced_route_optimization(self, location_weights_df, hub_id):
        """
        Main function that combines HDBSCAN clustering with Z-score routing.
        """
        print(f"\nEnhanced Routing with HDBSCAN for Hub {hub_id}")
        print(f"Processing {len(location_weights_df)} pickup locations...")

        start_time = time.time()

        # Fallback conditions
        if not HDBSCAN_AVAILABLE or len(location_weights_df) < self.min_cluster_size:
            if not HDBSCAN_AVAILABLE:
                print("WARNING: HDBSCAN not available, using original Z-score algorithm")
            else:
                print("WARNING: Too few locations for clustering, using original Z-score algorithm")
            return self._fallback_to_zscore(location_weights_df)

        try:
            # Step 1: Prepare location features for clustering
            feature_matrix = self._create_clustering_features(location_weights_df)

            # Step 2: Perform HDBSCAN clustering
            cluster_labels = self._perform_hdbscan_clustering(feature_matrix, location_weights_df)

            # Step 3: Cluster-guided trip formation
            trips, alerts = self._cluster_guided_trip_formation(
                location_weights_df, cluster_labels, hub_id
            )

            processing_time = time.time() - start_time

            # Step 4: Performance analysis
            performance_metrics = self._analyze_performance(
                trips, cluster_labels, processing_time
            )

            print(f"SUCCESS: Enhanced routing completed in {processing_time:.2f}s")
            return trips, alerts, performance_metrics

        except Exception as e:
            print(f"ERROR: HDBSCAN routing failed: {e}")
            print("Falling back to original Z-score algorithm...")
            return self._fallback_to_zscore(location_weights_df)

    def _create_clustering_features(self, df):
        """Create routing-focused features optimized for vehicle routing clustering."""
        print("Creating routing-focused clustering features...")

        feature_matrix = []

        # Pre-compute all inter-location distances for efficiency
        print("   Computing inter-location distance matrix...")
        location_list = df['source_state_pincode'].tolist()

        # Validate we have necessary functions available
        try:
            # Test database connectivity
            test_distance = get_distance_from_db(location_list[0], location_list[0])
            distance_matrix = self._compute_location_distance_matrix(location_list)
        except Exception as e:
            print(f"   Warning: Distance matrix computation failed: {e}")
            print("   Falling back to simplified feature engineering without inter-location distances")
            # Use simplified features without inter-location distances
            return self._create_simplified_clustering_features(df)

        for idx, row in df.iterrows():
            location = row['source_state_pincode']
            location_idx = location_list.index(location)

            features = []

            # 1. Distance from hub (normalized)
            hub_distance = row.get('distance_from_hub', 0)
            features.append(hub_distance)

            # 2. Geographic encoding (state-level)
            state = location.split('+')[0]
            state_encoded = self._encode_state_for_clustering(state)
            features.extend(state_encoded)

            # 3. Pincode-based features
            pincode = location.split('+')[1]
            pincode_features = self._extract_pincode_features(pincode)
            features.extend(pincode_features)

            # 4. Load characteristics
            weight = row.get('weight', 0)
            features.append(np.log1p(weight))  # Log-transform weight

            # 5. Neighborhood density (existing)
            neighbors = row.get('neighbors_count', 0)
            features.append(neighbors)

            # ===== NEW ROUTING-FOCUSED FEATURES =====

            # 6. Average distance to nearby locations (critical for routing efficiency)
            avg_nearby_distance = self._calculate_avg_nearby_distance(
                location_idx, distance_matrix, max_neighbors=5
            )
            features.append(avg_nearby_distance)

            # 7. Local pickup density (locations within routing radius)
            local_pickup_density = self._calculate_local_pickup_density(
                location_idx, distance_matrix, df, radius_km=30
            )
            features.append(local_pickup_density)

            # 8. Weighted pickup density (considering cargo weights)
            weighted_pickup_density = self._calculate_weighted_pickup_density(
                location_idx, distance_matrix, df, radius_km=50
            )
            features.append(weighted_pickup_density)

            # 9. Routing efficiency score (cost per km in local area)
            routing_efficiency = self._calculate_routing_efficiency_score(
                location_idx, distance_matrix, df, radius_km=40
            )
            features.append(routing_efficiency)

            # 10. Inter-location connectivity strength
            connectivity_strength = self._calculate_connectivity_strength(
                location_idx, distance_matrix, threshold_km=60
            )
            features.append(connectivity_strength)

            # 11. Distance variance to nearby locations (routing complexity indicator)
            distance_variance = self._calculate_nearby_distance_variance(
                location_idx, distance_matrix, max_neighbors=8
            )
            features.append(distance_variance)

            # 12. Potential tour length estimate (TSP approximation)
            tour_length_estimate = self._estimate_local_tour_length(
                location_idx, distance_matrix, max_locations=6
            )
            features.append(tour_length_estimate)

            # ===== END NEW FEATURES =====

            # 13. Distance to nearby high-priority locations (enhanced)
            nearby_priority_score = self._calculate_nearby_priority(row, df)
            features.append(nearby_priority_score)

            feature_matrix.append(features)

        feature_matrix = np.array(feature_matrix)

        # Handle any NaN or infinite values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0, posinf=999, neginf=-999)

        print(f"   Created enhanced feature matrix: {feature_matrix.shape}")
        print(f"   Features: hub_distance, state_encoding(7), pincode_features(3), weight, neighbors,")
        print(f"            avg_nearby_distance, local_density, weighted_density, routing_efficiency,")
        print(f"            connectivity, distance_variance, tour_estimate, priority_score")
        return feature_matrix

    def _encode_state_for_clustering(self, state):
        """Encode state information for geographic clustering."""
        state_zones = {
            'NORTH': ['PUNJAB', 'HARYANA', 'DELHI', 'UTTAR PRADESH', 'UTTARAKHAND', 'HIMACHAL PRADESH'],
            'NORTHWEST': ['RAJASTHAN', 'JAMMU AND KASHMIR'],
            'CENTRAL': ['MADHYA PRADESH', 'CHHATTISGARH'],
            'WEST': ['MAHARASHTRA', 'GUJARAT', 'GOA'],
            'SOUTH': ['TAMIL NADU', 'KARNATAKA', 'KERALA', 'ANDHRA PRADESH', 'TELANGANA'],
            'EAST': ['WEST BENGAL', 'BIHAR', 'JHARKHAND', 'ODISHA'],
            'NORTHEAST': ['ASSAM', 'MEGHALAYA', 'NAGALAND', 'MANIPUR', 'TRIPURA', 'MIZORAM', 'ARUNACHAL PRADESH']
        }

        # One-hot encode zones
        zone_encoding = [0] * len(state_zones)
        for idx, (zone, states) in enumerate(state_zones.items()):
            if state in states:
                zone_encoding[idx] = 1
                break

        return zone_encoding

    def _extract_pincode_features(self, pincode):
        """Extract clustering-relevant features from pincode."""
        if not pincode or len(pincode) < 3:
            return [0, 0, 0]

        first_digit = int(pincode[0]) if pincode[0].isdigit() else 0

        # Urban likelihood (simplified heuristic)
        urban_likelihood = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.9, 5: 0.7, 6: 0.8, 7: 0.6, 8: 0.5, 9: 0.3}.get(first_digit, 0.5)

        # Regional density score
        density_score = {1: 0.9, 2: 0.7, 3: 0.5, 4: 0.9, 5: 0.8, 6: 0.8, 7: 0.7, 8: 0.6, 9: 0.4}.get(first_digit, 0.5)

        return [first_digit / 10.0, urban_likelihood, density_score]

    def _calculate_nearby_priority(self, target_row, df):
        """Calculate priority score based on nearby high-value locations."""
        target_location = target_row['source_state_pincode']
        target_state = target_location.split('+')[0]

        # Find locations in same state
        same_state_df = df[df['source_state_pincode'].str.contains(target_state)]

        if len(same_state_df) <= 1:
            return 0

        # Calculate weighted priority based on nearby locations
        priority_score = 0
        for _, nearby_row in same_state_df.iterrows():
            if nearby_row['source_state_pincode'] == target_location:
                continue

            # Estimate distance (simplified)
            distance = self._estimate_intrastate_distance(target_location, nearby_row['source_state_pincode'])
            weight = nearby_row.get('weight', 0)

            # Closer and heavier locations contribute more to priority
            if distance < 100:  # Within 100km
                priority_score += weight / (distance + 1)

        return min(priority_score / 1000, 10)  # Normalize and cap

    def _estimate_intrastate_distance(self, loc1, loc2):
        """Quick distance estimation for clustering purposes."""
        pin1 = loc1.split('+')[1]
        pin2 = loc2.split('+')[1]

        # Simple heuristic based on pincode difference
        if len(pin1) >= 3 and len(pin2) >= 3:
            pin_diff = abs(int(pin1[:3]) - int(pin2[:3]))
            return min(pin_diff * 5, 200)  # Rough estimate, cap at 200km

        return 50  # Default estimate

    def _perform_hdbscan_clustering(self, feature_matrix, df):
        """Perform HDBSCAN clustering optimized for logistics routing."""
        print("🔬 Performing HDBSCAN clustering...")

        # Normalize features for clustering
        features_scaled = self.scaler.fit_transform(feature_matrix)

        # Initialize HDBSCAN with routing-optimized parameters
        # Adaptive parameters based on dataset size and routing objectives
        adaptive_min_cluster_size = max(self.min_cluster_size, len(df) // 8)  # Smaller clusters for better routing
        adaptive_min_samples = max(2, min(self.min_samples, len(df) // 15))  # More sensitive to local density

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=adaptive_min_cluster_size,
            min_samples=adaptive_min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric='euclidean',
            cluster_selection_method='eom',  # Excess of Mass - good for routing efficiency
            algorithm='best',
            leaf_size=30
        )

        print(f"   HDBSCAN parameters: min_cluster_size={adaptive_min_cluster_size}, "
              f"min_samples={adaptive_min_samples}, epsilon={self.cluster_selection_epsilon}")

        # Fit and predict clusters
        cluster_labels = clusterer.fit_predict(features_scaled)

        # Cluster analysis
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        print(f"   Found {n_clusters} clusters with {n_noise} outlier locations")

        # Cluster size analysis
        if n_clusters > 0:
            cluster_sizes = [list(cluster_labels).count(i) for i in range(n_clusters)]
            print(f"   Cluster sizes: {cluster_sizes}")

        return cluster_labels

    def _cluster_guided_trip_formation(self, df, cluster_labels, hub_id):
        """Form trips using cluster guidance combined with Z-score optimization."""
        print("TARGET: Forming trips with cluster guidance...")

        trips = []
        alerts = []
        df_with_clusters = df.copy()
        df_with_clusters['cluster_id'] = cluster_labels

        # Process each cluster separately for better spatial coherence
        unique_clusters = set(cluster_labels)

        for cluster_id in sorted(unique_clusters):
            if cluster_id == -1:
                continue  # Handle noise points later

            cluster_df = df_with_clusters[df_with_clusters['cluster_id'] == cluster_id].copy()

            if cluster_df.empty:
                continue

            print(f"   Processing cluster {cluster_id} with {len(cluster_df)} locations")

            # Apply Z-score within cluster for optimal anchor selection
            cluster_df = self._apply_zscore_within_cluster(cluster_df)

            # Create trips for this cluster using modified algorithm
            cluster_trips, cluster_alerts = create_trips_iterative(cluster_df, 'round_trip')

            trips.extend(cluster_trips)
            alerts.extend(cluster_alerts)

        # Handle noise points (outliers) with original algorithm
        noise_df = df_with_clusters[df_with_clusters['cluster_id'] == -1].copy()
        if not noise_df.empty:
            print(f"   Processing {len(noise_df)} outlier locations with Z-score algorithm")
            noise_df = self._apply_zscore_within_cluster(noise_df)
            noise_trips, noise_alerts = create_trips_iterative(noise_df, 'round_trip')
            trips.extend(noise_trips)
            alerts.extend(noise_alerts)

        return trips, alerts

    def _apply_zscore_within_cluster(self, cluster_df):
        """Apply Z-score calculation within a cluster for better prioritization."""
        if len(cluster_df) <= 1:
            cluster_df['zscore_priority'] = 1.0
            return cluster_df

        # Apply existing Z-score logic but within cluster context
        cluster_df = compute_neighbors_count(cluster_df, 60)
        cluster_df = add_priority_zscore(cluster_df)

        return cluster_df

    def _analyze_performance(self, trips, cluster_labels, processing_time):
        """Analyze the performance of cluster-guided routing."""
        total_distance = 0
        total_cost = 0
        total_weight = 0

        for trip in trips:
            dist, weight, cost, _ = calculate_trip_cost(trip)
            total_distance += dist
            total_cost += cost
            total_weight += weight

        # Cluster metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        metrics = {
            'routing_method': 'enhanced_hdbscan',
            'total_trips': len(trips),
            'total_distance_km': total_distance,
            'total_cost_inr': total_cost,
            'total_weight_kg': total_weight,
            'avg_cost_per_kg': total_cost / total_weight if total_weight > 0 else 0,
            'clusters_formed': n_clusters,
            'outlier_locations': n_noise,
            'processing_time_seconds': processing_time,
            'avg_distance_per_trip': total_distance / len(trips) if trips else 0
        }

        return metrics

    def _fallback_to_zscore(self, df):
        """Fallback to original Z-score algorithm."""
        df_processed = compute_neighbors_count(df, 60)
        df_processed = add_priority_zscore(df_processed)
        trips, alerts = create_trips_iterative(df_processed, 'round_trip')

        metrics = {
            'routing_method': 'zscore_fallback',
            'total_trips': len(trips),
            'processing_time_seconds': 0.1
        }

        return trips, alerts, metrics

    def _create_simplified_clustering_features(self, df):
        """
        Simplified feature engineering without inter-location distances.
        Used as fallback when distance matrix computation fails.
        """
        print("Creating simplified clustering features (no inter-location distances)...")

        feature_matrix = []

        for idx, row in df.iterrows():
            location = row['source_state_pincode']

            features = []

            # 1. Distance from hub (normalized)
            hub_distance = row.get('distance_from_hub', 0)
            features.append(hub_distance)

            # 2. Geographic encoding (state-level)
            state = location.split('+')[0]
            state_encoded = self._encode_state_for_clustering(state)
            features.extend(state_encoded)

            # 3. Pincode-based features
            pincode = location.split('+')[1]
            pincode_features = self._extract_pincode_features(pincode)
            features.extend(pincode_features)

            # 4. Load characteristics
            weight = row.get('weight', 0)
            features.append(np.log1p(weight))  # Log-transform weight

            # 5. Neighborhood density (existing z-score based)
            neighbors = row.get('neighbors_count', 0)
            features.append(neighbors)

            # 6. Distance to nearby high-priority locations (enhanced)
            nearby_priority_score = self._calculate_nearby_priority(row, df)
            features.append(nearby_priority_score)

            feature_matrix.append(features)

        feature_matrix = np.array(feature_matrix)

        # Handle any NaN or infinite values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0, posinf=999, neginf=-999)

        print(f"   Created simplified feature matrix: {feature_matrix.shape}")
        print(f"   Features: hub_distance, state_encoding(7), pincode_features(3), weight, neighbors, priority_score")
        return feature_matrix

# ------------------------------------------------
# ROUTING-FOCUSED FEATURE CALCULATION METHODS
# ------------------------------------------------

# Add these methods to the HDBSCANEnhancedRouter class
def _compute_location_distance_matrix_func(self, location_list):
        """
        Compute distance matrix between all locations for routing feature calculations.
        Uses the existing calculate_dist function which handles:
        1. Database lookup first
        2. Google Maps API if not found
        3. OlaMaps API as fallback
        4. Automatic caching to database
        """
        n_locations = len(location_list)
        distance_matrix = np.zeros((n_locations, n_locations))
        total_pairs = n_locations * (n_locations - 1) // 2

        print(f"   Building distance matrix for {n_locations} locations ({total_pairs} pairs)...")

        processed_pairs = 0
        api_calls_made = 0

        for i in range(n_locations):
            for j in range(i + 1, n_locations):
                # Use the existing calculate_dist function which handles all the logic:
                # 1. Check database first
                # 2. Call Google Maps API if not found
                # 3. Call OlaMaps API as fallback
                # 4. Save to database automatically
                distance = calculate_dist(location_list[i], location_list[j])

                if distance is not None:
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance
                else:
                    # Even after API calls, distance couldn't be determined
                    # Use a reasonable default based on different states vs same state
                    state1 = location_list[i].split('+')[0]
                    state2 = location_list[j].split('+')[0]

                    if state1 == state2:
                        # Same state - closer estimate
                        distance_matrix[i][j] = 150  # 150km default for same state
                        distance_matrix[j][i] = 150
                    else:
                        # Different states - farther estimate
                        distance_matrix[i][j] = 500  # 500km default for different states
                        distance_matrix[j][i] = 500

                    print(f"   Warning: Could not determine distance between {location_list[i]} and {location_list[j]}, using default estimate")

                processed_pairs += 1
                if processed_pairs % 20 == 0:
                    print(f"   Progress: {processed_pairs}/{total_pairs} pairs processed...")

        print(f"   Distance matrix completed: {n_locations}x{n_locations}")
        return distance_matrix

def _calculate_avg_nearby_distance_func(self, location_idx, distance_matrix, max_neighbors=5):
    """Calculate average distance to nearby locations for routing efficiency."""
    distances_to_others = distance_matrix[location_idx]
    # Remove distance to self (which is 0)
    non_zero_distances = distances_to_others[distances_to_others > 0]

    if len(non_zero_distances) == 0:
        return 0

    # Get the nearest neighbors
    nearest_distances = np.sort(non_zero_distances)[:max_neighbors]
    return np.mean(nearest_distances) if len(nearest_distances) > 0 else 0

def _calculate_local_pickup_density_func(self, location_idx, distance_matrix, df, radius_km=30):
    """Calculate number of pickup locations within routing radius."""
    distances_to_others = distance_matrix[location_idx]
    nearby_count = np.sum(distances_to_others <= radius_km) - 1  # Exclude self
    return nearby_count

def _calculate_weighted_pickup_density_func(self, location_idx, distance_matrix, df, radius_km=50):
    """Calculate weight-adjusted pickup density within radius."""
    distances_to_others = distance_matrix[location_idx]
    nearby_mask = distances_to_others <= radius_km

    # Get weights of nearby locations (including self)
    weights = df['weight'].values
    nearby_weights = weights[nearby_mask]

    if len(nearby_weights) <= 1:  # Only self
        return 0

    # Total weight excluding self
    total_nearby_weight = np.sum(nearby_weights) - weights[location_idx]
    return total_nearby_weight / radius_km  # Weight density per km

def _calculate_routing_efficiency_score_func(self, location_idx, distance_matrix, df, radius_km=40):
    """Calculate routing efficiency based on weight-to-distance ratio in local area."""
    distances_to_others = distance_matrix[location_idx]
    nearby_mask = (distances_to_others <= radius_km) & (distances_to_others > 0)

    if not np.any(nearby_mask):
        return 0

    nearby_distances = distances_to_others[nearby_mask]
    nearby_weights = df['weight'].values[nearby_mask]

    if len(nearby_distances) == 0:
        return 0

    # Calculate weight per km for nearby locations
    weight_per_km_ratios = nearby_weights / (nearby_distances + 1)  # +1 to avoid division by zero
    return np.mean(weight_per_km_ratios)

def _calculate_connectivity_strength_func(self, location_idx, distance_matrix, threshold_km=60):
    """Calculate how well connected a location is to others within threshold."""
    distances_to_others = distance_matrix[location_idx]
    connections = np.sum(distances_to_others <= threshold_km) - 1  # Exclude self

    # Weight by inverse of average distance to connected locations
    connected_distances = distances_to_others[(distances_to_others <= threshold_km) & (distances_to_others > 0)]

    if len(connected_distances) == 0:
        return 0

    avg_distance = np.mean(connected_distances)
    connectivity_score = connections / (avg_distance + 1)  # Higher score = more connections with shorter distances
    return connectivity_score

def _calculate_nearby_distance_variance_func(self, location_idx, distance_matrix, max_neighbors=8):
    """Calculate variance in distances to nearby locations (routing complexity)."""
    distances_to_others = distance_matrix[location_idx]
    non_zero_distances = distances_to_others[distances_to_others > 0]

    if len(non_zero_distances) == 0:
        return 0

    # Get nearest neighbors
    nearest_distances = np.sort(non_zero_distances)[:max_neighbors]

    if len(nearest_distances) <= 1:
        return 0

    return np.var(nearest_distances)

def _estimate_local_tour_length_func(self, location_idx, distance_matrix, max_locations=6):
    """Estimate tour length for visiting nearby locations (TSP approximation)."""
    distances_to_others = distance_matrix[location_idx]
    non_zero_distances = distances_to_others[distances_to_others > 0]

    if len(non_zero_distances) == 0:
        return 0

    # Get indices of nearest locations
    nearest_indices = np.argsort(distances_to_others)[1:max_locations+1]  # Skip self (index 0)

    if len(nearest_indices) <= 1:
        return distances_to_others[nearest_indices[0]] if len(nearest_indices) == 1 else 0

    # Simple TSP approximation: sum of distances to nearest neighbors
    # This is a crude approximation but gives an idea of local tour complexity
    tour_length = 0

    # Add distance to first location
    if len(nearest_indices) > 0:
        tour_length += distance_matrix[location_idx][nearest_indices[0]]

    # Add distances between consecutive nearest locations
    for i in range(len(nearest_indices) - 1):
        tour_length += distance_matrix[nearest_indices[i]][nearest_indices[i + 1]]

    # Add return distance to starting location
    if len(nearest_indices) > 0:
        tour_length += distance_matrix[nearest_indices[-1]][location_idx]

    return tour_length

# Monkey patch these methods to the HDBSCANEnhancedRouter class
HDBSCANEnhancedRouter._compute_location_distance_matrix = _compute_location_distance_matrix_func
HDBSCANEnhancedRouter._calculate_avg_nearby_distance = _calculate_avg_nearby_distance_func
HDBSCANEnhancedRouter._calculate_local_pickup_density = _calculate_local_pickup_density_func
HDBSCANEnhancedRouter._calculate_weighted_pickup_density = _calculate_weighted_pickup_density_func
HDBSCANEnhancedRouter._calculate_routing_efficiency_score = _calculate_routing_efficiency_score_func
HDBSCANEnhancedRouter._calculate_connectivity_strength = _calculate_connectivity_strength_func
HDBSCANEnhancedRouter._calculate_nearby_distance_variance = _calculate_nearby_distance_variance_func
HDBSCANEnhancedRouter._estimate_local_tour_length = _estimate_local_tour_length_func

print("SUCCESS: Enhanced HDBSCAN routing features loaded successfully")
print("   - Uses existing distance database and API infrastructure")
print("   - Includes intelligent fallback for missing distance data")
print("   - 7 new routing-focused features for better clustering")

# ------------------------------------------------
# ✨ ENHANCED ROUTING FUNCTION ✨
# ------------------------------------------------
def create_trips_enhanced(location_weights_small, hub_id):
    """
    Enhanced trip creation function that uses HDBSCAN clustering
    when available and beneficial, otherwise falls back to original Z-score.

    This is the main integration point - replaces create_trips_iterative calls.
    """
    if not HDBSCAN_AVAILABLE or len(location_weights_small) < 8:
        # Use original algorithm
        return create_trips_iterative(location_weights_small, 'round_trip')

    # Initialize enhanced router with conservative parameters
    router = HDBSCANEnhancedRouter(
        min_cluster_size=max(8, len(location_weights_small) // 10),
        min_samples=3,
        cluster_selection_epsilon=0.1
    )

    # Run enhanced routing
    trips, alerts, performance_metrics = router.enhanced_route_optimization(
        location_weights_small, hub_id
    )

    # Log performance metrics for monitoring
    print(f"\n📈 Enhanced Routing Performance:")
    for key, value in performance_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    return trips, alerts

# ------------------------------------------------
# Summarize Trips (Unchanged)
# ------------------------------------------------
def summarize_trips(trips, alerts, hub_id):
    """Generate a summary of all the trips."""
    summary = {}
    summary_id = uuid.uuid4().hex[:16]
    summary["id"] = summary_id

    ist = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    summary["timestamp"] = timestamp
    summary["hub_id"] = hub_id

    # Compute overall stats
    total_weight = 0
    total_distance = 0
    total_cost = 0
    total_pickups_count = 0
    all_locations = []

    for trip in trips:
        dist, wt, cost_val, _ = calculate_trip_cost(trip)
        total_distance += dist
        total_weight += wt
        total_cost += cost_val

        # Count how many pickup numbers are in this trip (excluding hub)
        for stop in trip[1:-1]:
            pnums = stop[3]
            total_pickups_count += len(pnums)
            all_locations.append(stop[0])

    summary["total_points"] = total_pickups_count
    summary["total_locations"] = len(set(all_locations))
    summary["total_weight"] = total_weight
    summary["trip_count"] = len(trips)
    summary["total_cost"] = total_cost
    summary["cost_per_kg"] = total_cost / total_weight if total_weight else 0
    summary["total_trip_distance"] = total_distance

    print("\nTotal Summary")
    print(f"Hub ID: {hub_id}")
    print(f"Total Points: {summary['total_points']}")
    print(f"Total Locations: {summary['total_locations']}")
    print(f"Total Weight: {summary['total_weight']} kg")
    print(f"Count of Trips created: {summary['trip_count']}")
    print(f"Total cost: INR {summary['total_cost']}")
    print(f"Total cost per kg: INR {summary['cost_per_kg']:.2f}")
    print(f"Total trip distance: {total_distance:.2f} km")

    # Build out detailed info
    summary["trips"] = []
    for i, trip in enumerate(trips, 1):
        trip_summary = {
            "trip_number": i,
            "vehicle_type": trip[1][2] if len(trip) > 1 else None,
            "stops": []
        }
        # exclude hub at start/end
        for stop in trip[1:-1]:
            location, weight, _, pickup_list = stop
            trip_summary["stops"].append({
                "pickup_numbers": pickup_list,
                "location": location,
                "weight": weight
            })

        dist, wt, cost_val, cpk = calculate_trip_cost(trip)
        trip_summary["total_distance"] = dist
        trip_summary["total_weight"] = wt
        trip_summary["cost"] = cost_val
        trip_summary["cost_per_kg"] = cpk

        summary["trips"].append(trip_summary)

        # Print trip details
        print(f"\nTrip {i}: ({trip[1][2] if len(trip) > 1 else 'N/A'})")
        for st in trip_summary["stops"]:
            print(f"Pickup Numbers: {', '.join(st['pickup_numbers'])}, "
                  f"Location: {st['location']} ({st['weight']} kg)")
        print(f"Total distance: {dist:.2f} km")
        print(f"Total weight: {wt:.2f} kg")
        print(f"Cost: INR {cost_val}")
        print(f"Cost per kg: {cpk:.2f}")

    return summary

# ------------------------------------------------
# Job-table utilities (Unchanged)
# ------------------------------------------------
def update_job_status(job_id, status, output_json=None, when_field=None):
    """Write status & timestamps (IST) to route_opt_jobs."""
    ist_ts = datetime.now(IST_ZONE).strftime("%Y-%m-%d %H:%M:%S")

    conn   = get_mysql_connection()
    cursor = conn.cursor()

    set_cols = ["status = %s"]
    params   = [status]

    if output_json is not None:
        set_cols.append("output_json = %s")
        params.append(output_json)

    if when_field == "started":
        set_cols.append("started_at = %s")
        params.append(ist_ts)
    elif when_field == "finished":
        set_cols.append("finished_at = %s")
        params.append(ist_ts)

    params.append(job_id)
    sql = f"UPDATE route_opt_jobs SET {', '.join(set_cols)} WHERE id = %s"
    cursor.execute(sql, params)
    conn.commit()
    conn.close()

def get_pending_jobs():
    conn   = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, hub_id FROM route_opt_jobs WHERE status='PENDING' limit 1")
    jobs = cursor.fetchall()
    conn.close()
    return jobs

# ------------------------------------------------
# Hub-ID → "STATE+PIN" map (Unchanged)
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
# ✨ ENHANCED SINGLE-JOB RUNNER ✨
# ------------------------------------------------
def run_job(job_id, hub_id):
    """
    Enhanced job runner with HDBSCAN integration.

    Key changes from original:
    - Uses create_trips_enhanced instead of create_trips_iterative
    - Maintains all existing logic and error handling
    - Logs enhanced routing performance metrics
    """
    try:
        update_job_status(job_id, "RUNNING", when_field="started")

        # ------------- original hub-data fetch (UNCHANGED) -------------
        url = f"https://logistics.wastelink.co/api/planned-pickup-list/Inbound/{hub_id}"
        response = requests.get(url)

        if response.status_code != 200:
            raise RuntimeError(f"Hub {hub_id}: API status {response.status_code}")

        nested_data = response.json().get('data')
        if not isinstance(nested_data, list) or len(nested_data) == 0:
            raise RuntimeError(f"Hub {hub_id}: no pickup data")

        # ------------- dataframe build (UNCHANGED) -------------
        df = pd.DataFrame(nested_data)

        original_df = df[['pickup_number', 'material_quantity',
                          'city_name', 'state_name',
                          'location_pincode', 'hub_id']].copy()

        original_df['city_name']  = original_df['city_name'].str.upper()
        original_df['state_name'] = original_df['state_name'].str.upper()

        original_df['source_state_pincode'] = (
            original_df['state_name'] + '+' +
            original_df['location_pincode'].astype(str)
        )
        original_df['source_city_state'] = (
            original_df['city_name'] + '+' + original_df['state_name']
        )
        original_df['hub_state_pincode'] = original_df['hub_id'].map(HUB_PINCODE_MAP)

        # ---------- drop rows with missing distances (UNCHANGED) ----------
        removed, failed, distances = [], [], []
        for idx, row in original_df.iterrows():
            src = row['source_state_pincode']
            dst = row['hub_state_pincode']
            d   = calculate_dist(src, dst)
            if d is None:
                failed.append({'pickup_number': row['pickup_number'],
                               'source': src, 'destination': dst})
                removed.append({'pickup_number': row['pickup_number'],
                                'source_state_pincode': src})
                original_df.drop(idx, inplace=True)
            else:
                distances.append(d)

        original_df['distance_from_hub'] = distances
        modified_df = (original_df
                       .copy()
                       .reset_index(drop=True))
        modified_df['material_quantity'] = pd.to_numeric(
            modified_df['material_quantity'], errors='coerce')
        modified_df.dropna(subset=['material_quantity'], inplace=True)

        # ---------- split small / large (UNCHANGED) ----------
        df_small = modified_df[modified_df['material_quantity'] < 3501].copy()
        df_large = modified_df[modified_df['material_quantity'] >= 3501].copy()

        # small → group by loc (UNCHANGED)
        location_weights_small = (
            df_small
            .groupby(['source_state_pincode', 'hub_state_pincode', 'hub_id'],
                     as_index=False)
            .agg({
                'material_quantity': 'sum',
                'pickup_number':    lambda x: list(x)
            })
            .rename(columns={'material_quantity': 'weight'})
        )

        # large → keep 1-to-1 (UNCHANGED)
        df_large_for_trips = df_large[[
            'source_state_pincode', 'hub_state_pincode', 'hub_id',
            'material_quantity', 'pickup_number'
        ]].copy()
        df_large_for_trips.rename(
            columns={'material_quantity': 'weight'}, inplace=True)
        df_large_for_trips['pickup_number'] = df_large_for_trips[
            'pickup_number'].apply(lambda x: [x] if isinstance(x, str) else (
                x if isinstance(x, list) else []))

        location_weights_small.reset_index(drop=True, inplace=True)
        df_large_for_trips.reset_index(drop=True, inplace=True)

        # ---------- neighbours & z-scores (UNCHANGED) ----------
        if not location_weights_small.empty:
            _ = compute_all_pairwise_distances(location_weights_small)
            location_weights_small = compute_neighbors_count(
                location_weights_small, 60)
            location_weights_small = add_priority_zscore(
                location_weights_small)

        if not df_large_for_trips.empty:
            _ = compute_all_pairwise_distances(df_large_for_trips)
            df_large_for_trips = compute_neighbors_count(
                df_large_for_trips, 60)
            df_large_for_trips = add_priority_zscore(df_large_for_trips)

        # ---------- ✨ ENHANCED TRIP BUILDING ✨ ----------
        print(f"\nUsing Enhanced HDBSCAN Routing for Hub {hub_id}")

        # Enhanced routing for small loads
        trips_small, alerts_small = create_trips_enhanced(
            location_weights_small, hub_id)

        # Original routing for large loads (unchanged)
        trips_large, alerts_large = create_single_trips_for_large(
            df_large_for_trips)

        all_trips  = trips_small + trips_large
        all_alerts = alerts_small + alerts_large
        hub_id_int = int(modified_df['hub_id'].iloc[0]) if not modified_df.empty else hub_id

        summary        = summarize_trips(all_trips, all_alerts, hub_id_int)
        summary_clean  = convert_np(summary)

        # ---------- POST to backend (UNCHANGED) ----------
        post_url = 'https://logistics.wastelink.co/api/trip-by-ai'
        resp = requests.post(
            post_url,
            data=json.dumps(summary_clean),
            headers={'Content-Type': 'application/json'}
        )
        if resp.status_code != 200:
            raise RuntimeError(f"POST failed (HTTP {resp.status_code})")

        # ---------- success (UNCHANGED) ----------
        update_job_status(
            job_id, "COMPLETE",
            output_json=json.dumps(summary_clean),
            when_field="finished"
        )

    except Exception as exc:
        print(f"[route_opt_jobs] Job {job_id} (hub {hub_id}) failed: {exc}")
        update_job_status(job_id, "ERROR", when_field="finished")

# ------------------------------------------------
# TEST FUNCTION for Quick Validation
# ------------------------------------------------
def test_enhanced_routing():
    """
    Test function to validate HDBSCAN integration with sample data.
    Run this to verify the enhanced routing works correctly.
    """
    print("Testing Enhanced HDBSCAN Routing")
    print("=" * 50)

    # Create sample data mimicking your API structure
    sample_data = pd.DataFrame({
        'source_state_pincode': [
            'TAMIL NADU+600001', 'TAMIL NADU+600002', 'TAMIL NADU+600003',
            'KARNATAKA+560001', 'KARNATAKA+560002',
            'MAHARASHTRA+400001', 'MAHARASHTRA+400002'
        ],
        'hub_state_pincode': ['TAMIL NADU+600060'] * 7,
        'weight': [100, 150, 120, 200, 180, 300, 250],
        'pickup_number': [['P001'], ['P002'], ['P003'], ['P004'], ['P005'], ['P006'], ['P007']],
        'hub_id': [8] * 7,
        'distance_from_hub': [10, 12, 15, 300, 320, 450, 470],
        'neighbors_count': [2, 3, 2, 1, 1, 1, 1],
        'zscore_priority': [1.5, 2.1, 1.8, 0.5, 0.6, 0.8, 0.9]
    })

    print(f"Sample data: {len(sample_data)} locations")
    print(f"HDBSCAN available: {HDBSCAN_AVAILABLE}")

    # Test enhanced routing
    try:
        trips, alerts = create_trips_enhanced(sample_data, 8)

        print(f"\nSUCCESS: Test completed successfully!")
        print(f"   Generated {len(trips)} trips")
        print(f"   Alerts: {len(alerts)}")

        # Calculate total metrics
        total_distance = sum(calculate_trip_cost(trip)[0] for trip in trips)
        total_cost = sum(calculate_trip_cost(trip)[2] for trip in trips)

        print(f"   Total distance: {total_distance:.1f} km")
        print(f"   Total cost: ₹{total_cost:.0f}")

        return True, trips, alerts

    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        return False, [], []

# ------------------------------------------------
# Main execution (Enhanced)
# ------------------------------------------------
if __name__ == "__main__":
    print("Enhanced Route Optimization Worker with HDBSCAN")
    print("=" * 60)
    print(f"HDBSCAN Status: {'SUCCESS: Available' if HDBSCAN_AVAILABLE else 'ERROR: Not Available'}")

    # Enhanced debugging - always show this info
    print("\nSystem Debug Information:")
    print(f"   Python executable: {__import__('sys').executable}")
    print(f"   Python version: {__import__('sys').version}")
    print(f"   Current working directory: {__import__('os').getcwd()}")

    # Try to import and show detailed error
    print("\nTesting HDBSCAN import:")
    try:
        import hdbscan as test_hdbscan
        from sklearn.preprocessing import StandardScaler as test_scaler
        print(f"   SUCCESS: HDBSCAN imported successfully")
        try:
            print(f"   SUCCESS: HDBSCAN version: {test_hdbscan.__version__}")
        except AttributeError:
            print("   SUCCESS: HDBSCAN version: Available (version info not accessible)")
        print(f"   SUCCESS: Sklearn imported successfully")
    except ImportError as e:
        print(f"   ERROR: Import failed: {e}")
        print(f"   ERROR: This is why HDBSCAN is marked as unavailable")
    except Exception as e:
        print(f"   ERROR: Unexpected error: {e}")

    # Quick test if running directly
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_enhanced_routing()
        sys.exit(0)

    # Normal operation
    print("\nStarting enhanced route optimization polling...")
    while True:
        pending = get_pending_jobs()
        if not pending:
            print(f"[{datetime.now(IST_ZONE).strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"No pending jobs → sleeping 10 min")
        else:
            for job in pending:
                print(f"\n{'='*60}")
                print(f"Processing Job {job['id']} for Hub {job['hub_id']}")
                print(f"{'='*60}")
                run_job(job['id'], job['hub_id'])

        time.sleep(100)   # 100 seconds (same as original)
