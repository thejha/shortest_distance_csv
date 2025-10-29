#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------
# Imports
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
import time                      # <-- for 10-minute sleep loop

pd.options.mode.chained_assignment = None  # default='warn'

# ------------------------------------------------
# API clients & globals
# ------------------------------------------------
gmaps = googlemaps.Client(key='AIzaSyDMtTPCRVzDEyv-rcwk6BNIWZi4-bI-WZo')

new_api_key = 'UApDWhcNIesmLCt65ZmpWCskYvEWUg1PxqwCaXhn'
request_id  = str(uuid.uuid4())
BATCH_SIZE  = 10                              # Google batch size
IST_ZONE    = pytz.timezone("Asia/Kolkata")

# ------------------------------------------------
# MySQL connection helper
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
# Google & OlaMaps utilities
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

# ------------------------------------------------
# Helper for range limit
# ------------------------------------------------
def exceeds_range(total_dist, vehicle_name):
    return total_dist > VEHICLES[vehicle_name]['range_km']

# ------------------------------------------------
# Trip-building, TSP, Z-score etc. (UNCHANGED)
# ------------------------------------------------
#   – sequence_locations
#   – compute_all_pairwise_distances
#   – compute_neighbors_count
#   – add_priority_zscore
#   – calculate_trip_cost
#   – build_trip_with_vehicle
#   – create_trips_iterative
#   – select_one_way_vehicle_for_large
#   – create_single_trips_for_large
#   – summarize_trips
#   >>> KEEP THE IDENTICAL DEFINITIONS FROM YOUR EXISTING SCRIPT <<<
# ------------------------------------------------
# (For brevity they’re omitted here but copy-paste them verbatim.)
# ------------------------------------------------

# ------------------------------------------------
# TSP / Reorder Logic
# ------------------------------------------------
def sequence_locations(trip, calculate_dist):
    """
    Optimize the sequence of locations in a trip to minimize total distance (Held-Karp TSP).
    trip[i] is a 4-tuple: (location, weight, vehicle_type, pickup_list)
    We'll reorder the middle portion (trip[1:-1]) in a TSP-like approach.
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

# ------------------------------------------------------------------
# Cost calculation  (✱ min_cost logic injected ✱)
# ------------------------------------------------------------------
def calculate_trip_cost(trip):
    """
    Returns:
        total_distance (km), total_weight (kg), final_cost (INR), cost_per_kg (INR/kg)
    The final_cost will never be lower than VEHICLES[vtype]['min_cost'].
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

    # --- accumulate distance & weight exactly as before ---
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

    # --- compute cost ---
    base_cost  = total_distance * specs["cost_per_km"]
    min_cost   = specs.get("min_cost", 0)
    final_cost = max(base_cost, min_cost)   # <-- enforce minimum charge

    return (
        total_distance,
        total_weight,
        final_cost,
        (final_cost / total_weight) if total_weight else 0
    )

# ------------------------------------------------
# FOR SMALL LOADS => ITERATIVE TRIPS (SAME AS BEFORE)
# ------------------------------------------------
def sort_vehicles_by_capacity_desc(vehicle_type_filter):
    """
    Return the list of vehicle names (with matching cost_type)
    sorted by capacity in DESC order.
    vehicle_type_filter is either 'round_trip' or 'one_way'
    """
    filtered = {
        v: specs for v, specs in VEHICLES.items()
        if specs['cost_type'] == vehicle_type_filter
    }
    # Sort by capacity descending
    sorted_list = sorted(filtered.keys(), key=lambda x: filtered[x]['capacity'], reverse=True)
    return sorted_list

# ------------------------------------------------------------------
# TRIP‑BUILDER (round‑trip fleet) – range guard injected
# ------------------------------------------------------------------
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

    # ✱ RANGE CHECK ✱
    td, _, _, _ = calculate_trip_cost(candidate_trip)
    if exceeds_range(td, vehicle_name):
        return None, []

    return candidate_trip, used_indices

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
# ### NEW ###: For LARGE single-point loads => pick nearest capacity one-way vehicle
# ------------------------------------------------
def select_one_way_vehicle_for_large(load, hub_id):
    """
    Return the one-way vehicle whose capacity >= load and
    is as close to 'load' as possible (the smallest capacity that still fits).
    Must also check hub availability.
    """
    # Filter only vehicles that are one_way, serve this hub, and have capacity >= load
    candidates = []
    for v_name, v_specs in VEHICLES.items():
        if v_specs['cost_type'] == 'one_way' and hub_id in v_specs['hub_id'] and v_specs['capacity'] >= load:
            candidates.append((v_name, v_specs['capacity']))

    if not candidates:
        return None  # no suitable one-way vehicle

    # sort by capacity ascending
    candidates.sort(key=lambda x: x[1])
    # pick the first => smallest capacity that fits
    return candidates[0][0]

# ------------------------------------------------------------------
# ONE‑WAY large‑load builder – range guard injected
# ------------------------------------------------------------------
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
        if exceeds_range(td, trip[1][2]):       #  trip[1][2] is vehicle name
            alerts.append(
                f"Trip for {loc} exceeds {v_name} range "
                f"({td:.1f} km > {VEHICLES[v_name]['range_km']} km)"
            )
            continue          # skip trip entirely (or remove 'continue' to keep)

        trips.append(trip)
    return trips, alerts
# ------------------------------------------------
# Summarize Trips
# ------------------------------------------------
def summarize_trips(trips, alerts, hub_id):
    """
    Generate a summary of all the trips.

    NOTE: Each trip stop is a tuple of:
      (location, weight, vehicle_type, pickup_numbers_list)
    """
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
# Job-table utilities (NEW / REPLACED)
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
# Hub-ID → "STATE+PIN" map (needed inside run_job)
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
# Single-job runner (wraps the old hub loop)
# ------------------------------------------------
def run_job(job_id, hub_id):
    try:
        update_job_status(job_id, "RUNNING", when_field="started")

        # ------------- original hub-data fetch -------------
        url = f"https://logistics.wastelink.co/api/planned-pickup-list/Inbound/{hub_id}"
        response = requests.get(url)

        if response.status_code != 200:
            raise RuntimeError(f"Hub {hub_id}: API status {response.status_code}")

        nested_data = response.json().get('data')
        if not isinstance(nested_data, list) or len(nested_data) == 0:
            raise RuntimeError(f"Hub {hub_id}: no pickup data")

        # ------------- dataframe build (unchanged) -------------
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

        # ---------- drop rows with missing distances ----------
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

        # ---------- split small / large ----------
        df_small = modified_df[modified_df['material_quantity'] < 3501].copy()
        df_large = modified_df[modified_df['material_quantity'] >= 3501].copy()

        # small → group by loc
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

        # large → keep 1-to-1
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

        # ---------- neighbours & z-scores ----------
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

        # ---------- build trips ----------
        trips_small, alerts_small = create_trips_iterative(
            location_weights_small, vehicle_type_filter='round_trip')
        trips_large, alerts_large = create_single_trips_for_large(
            df_large_for_trips)

        all_trips  = trips_small + trips_large
        all_alerts = alerts_small + alerts_large
        hub_id_int = int(modified_df['hub_id'].iloc[0]) if not modified_df.empty else hub_id

        summary        = summarize_trips(all_trips, all_alerts, hub_id_int)
        summary_clean  = convert_np(summary)

        # ---------- POST to backend ----------
        post_url = 'https://logistics.wastelink.co/api/trip-by-ai'
        resp = requests.post(
            post_url,
            data=json.dumps(summary_clean),
            headers={'Content-Type': 'application/json'}
        )
        if resp.status_code != 200:
            raise RuntimeError(f"POST failed (HTTP {resp.status_code})")

        # ---------- success ----------
        update_job_status(
            job_id, "COMPLETE",
            output_json=json.dumps(summary_clean),
            when_field="finished"
        )

    except Exception as exc:
        print(f"[route_opt_jobs] Job {job_id} (hub {hub_id}) failed: {exc}")
        update_job_status(job_id, "ERROR", when_field="finished")


# ------------------------------------------------
# Poller – run every 10 minutes
# ------------------------------------------------
if __name__ == "__main__":
    while True:
        pending = get_pending_jobs()
        if not pending:
            print(f"[{datetime.now(IST_ZONE).strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"No pending jobs → sleeping 10 min")
        else:
            for job in pending:
                run_job(job['id'], job['hub_id'])

        time.sleep(100)   # 10 min
