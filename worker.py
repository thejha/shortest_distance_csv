import time
import json
import traceback
import itertools
from datetime import datetime

import mysql.connector

# -------------------------------------------------------------------------
# 1) DB + Distance Functions (like your existing code)
# -------------------------------------------------------------------------
GOOGLE_API_KEY = "AIzaSyDMtTPCRVzDEyv-rcwk6BNIWZi4-bI-WZo"

def get_mysql_connection():
    return mysql.connector.connect(
        host='vaibhavjha.mysql.pythonanywhere-services.com',
        user='vaibhavjha',
        password='anubhav21',
        database='vaibhavjha$database'
    )

def get_distance_from_db(source, destination):
    conn = get_mysql_connection()
    cursor = conn.cursor()
    query = '''
        SELECT distance FROM distances
        WHERE source = %s AND destination = %s
    '''
    cursor.execute(query, (source, destination))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def save_distance_to_db(source, destination, distance,
                        source_geocode=None, destination_geocode=None):
    conn = get_mysql_connection()
    cursor = conn.cursor()
    query = '''
        INSERT INTO distances
            (source, destination, distance, timestamp, source_geocode, destination_geocode)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            distance = VALUES(distance),
            timestamp = VALUES(timestamp),
            source_geocode = VALUES(source_geocode),
            destination_geocode = VALUES(destination_geocode)
    '''
    cursor.execute(query, (
        source,
        destination,
        distance,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        source_geocode,
        destination_geocode
    ))
    conn.commit()
    conn.close()

def geocode_address(address):
    """(Optional) Convert address to lat,lng using Google Geocoding."""
    import requests
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": GOOGLE_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data["status"] == "OK":
            loc = data["results"][0]["geometry"]["location"]
            return f"{loc['lat']},{loc['lng']}"
    except Exception as e:
        print(f"Geocoding error for '{address}': {e}")
    return None

def get_distance_from_google(source, destination, src_geo=None, dst_geo=None):
    """Use Google Distance Matrix to get distance in km."""
    import requests

    origin = src_geo if src_geo else source
    dest   = dst_geo if dst_geo else destination

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": dest,
        "key": GOOGLE_API_KEY,
        "units": "metric"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data["status"] == "OK":
            element = data["rows"][0]["elements"][0]
            if element["status"] == "OK":
                dist_meters = element["distance"]["value"]
                return dist_meters / 1000.0
    except Exception as e:
        print(f"Distance matrix error: {e}")
    return None

def calculate_dist(source, destination):
    """
    1) Check DB
    2) If not found, do geocode -> distance matrix
    3) Save in DB
    4) Return distance in km (or None)
    """
    dist_db = get_distance_from_db(source, destination)
    if dist_db is not None:
        return dist_db

    # Not in DB => geocode addresses
    src_geo = geocode_address(source)
    dst_geo = geocode_address(destination)
    dist_google = get_distance_from_google(source, destination, src_geo, dst_geo)
    if dist_google is not None:
        save_distance_to_db(source, destination, dist_google, src_geo, dst_geo)
        return dist_google

    return None

# -------------------------------------------------------------------------
# 2) Held-Karp TSP
# -------------------------------------------------------------------------
def sequence_locations(trip, calc_dist):
    """
    trip is a list of (location_str, pickup_id).
    We'll reorder trip[1:-1] using Held-Karp.
    trip[0] and trip[-1] are the same 'hub'.
    """
    if len(trip) <= 2:
        return trip

    middle = trip[1:-1]
    if len(middle) <= 1:
        return trip

    n = len(middle)
    # Distances dictionary: dist[(i,j)], where 0 => hub, 1..n => each middle item
    dist = {}

    # Fill distance among middle points
    for i, loc_i in enumerate(middle, start=1):
        for j, loc_j in enumerate(middle, start=1):
            if i != j:
                dval = calc_dist(loc_i[0], loc_j[0])
                dist[(i, j)] = dval if dval is not None else float('inf')

    # Distances from hub(0)->i and i->hub(0)
    hub_start = trip[0][0]
    hub_end   = trip[-1][0]
    for i, loc_i in enumerate(middle, start=1):
        d_hub_to = calc_dist(hub_start, loc_i[0])
        d_to_hub = calc_dist(loc_i[0], hub_end)
        dist[(0, i)] = d_hub_to if d_hub_to is not None else float('inf')
        dist[(i, 0)] = d_to_hub if d_to_hub is not None else float('inf')

    # Held-Karp DP
    dp = {}
    parent = {}

    # Subset size 1
    for i in range(1, n+1):
        dp[(1 << (i-1), i)] = dist.get((0, i), float('inf'))

    import math

    for subset_size in range(2, n+1):
        for subset in itertools.combinations(range(1, n+1), subset_size):
            subset_mask = 0
            for s in subset:
                subset_mask |= (1 << (s-1))
            for next_loc in subset:
                prev_mask = subset_mask & ~(1 << (next_loc-1))
                dp_key = (subset_mask, next_loc)
                if dp_key not in dp:
                    dp[dp_key] = math.inf
                min_cost = math.inf
                min_prev = None
                for prev_loc in subset:
                    if prev_loc != next_loc:
                        prev_dp_key = (prev_mask, prev_loc)
                        cost_prev_next = dist.get((prev_loc, next_loc), math.inf)
                        if prev_dp_key in dp and cost_prev_next < math.inf:
                            candidate = dp[prev_dp_key] + cost_prev_next
                            if candidate < min_cost:
                                min_cost = candidate
                                min_prev = prev_loc
                dp[dp_key] = min_cost
                if min_cost < math.inf:
                    parent[dp_key] = min_prev

    final_mask = (1 << n) - 1
    best_cost = float('inf')
    best_last = None
    for i in range(1, n+1):
        dp_key = (final_mask, i)
        d_i_to_0 = dist.get((i, 0), float('inf'))
        if dp_key in dp and d_i_to_0 < float('inf'):
            total_cost = dp[dp_key] + d_i_to_0
            if total_cost < best_cost:
                best_cost = total_cost
                best_last = i

    if best_last is None:
        # No feasible route
        return trip

    # Reconstruct path
    best_path = [best_last]
    curr_node = best_last
    curr_mask = final_mask
    while True:
        prev_node = parent.get((curr_mask, curr_node))
        if prev_node is None:
            break
        best_path.append(prev_node)
        curr_mask &= ~(1 << (curr_node - 1))
        curr_node = prev_node
    best_path.reverse()

    new_trip = [trip[0]]
    for idx in best_path:
        new_trip.append(middle[idx - 1])
    new_trip.append(trip[-1])
    return new_trip

# -------------------------------------------------------------------------
# 3) Process TSP Job
# -------------------------------------------------------------------------
hub_state_pincode_dict = {
    8:  "TAMIL NADU+600060",
    1:  "KARNATAKA+562114",
    2:  "HARYANA+131029",
    3:  "UTTAR PRADESH+226008",
    4:  "MAHARASHTRA+421506",
    26: "KARNATAKA+580024",
    5:  "TELANGANA+500070",
    18: "MAHARASHTRA+421506",
    28: "ANDHRA PRADESH+522001",
    13: "RAJASTHAN+302013",
    29:  "PUNJAB+141003",
    10: "UTTAR PRADESH+243302",
    20: "JAMMU AND KASHMIR+192301",
    22: "MADHYA PRADESH+462039",
    19: "UTTAR PRADESH+273403",
    14: "WEST BENGAL+712310",
    23: "BIHAR+800023",
    16: "UTTAR PRADESH+221108"
}

def process_tsp_job(input_data):
    """
    1) Extract pickups, hub_id, etc. from input_data
    2) Build trip array
    3) Run Held-Karp
    4) Calculate roundtrip + oneway distance
    5) Return a dict with results
    """

    pickup_ids = input_data["pickup_ids"]
    pickup_locs = input_data["pickup_locations"]
    hub_id = input_data["hub_id"]
    source_location = input_data.get("source_location", "")

    hub_address = hub_state_pincode_dict.get(hub_id)
    if not hub_address:
        raise ValueError(f"Invalid hub_id: {hub_id}")

    # Build trip
    # Start => (hub_address, "HUB_START")
    # Middle => list of (loc, pid)
    # End   => (hub_address, "HUB_END")
    start_hub = (hub_address, "HUB_START")
    end_hub   = (hub_address, "HUB_END")
    middle = [(loc, pid) for loc, pid in zip(pickup_locs, pickup_ids)]
    raw_trip = [start_hub] + middle + [end_hub]

    # Held-Karp
    optimized_trip = sequence_locations(raw_trip, calculate_dist)

    # Extract final pickup sequence
    optimized_middle = optimized_trip[1:-1]
    final_sequence = [item[1] for item in optimized_middle]

    # Compute roundtrip distance
    roundtrip = 0.0
    for i in range(len(optimized_trip) - 1):
        dval = calculate_dist(optimized_trip[i][0], optimized_trip[i+1][0])
        if dval is not None:
            roundtrip += dval

    # Compute oneway distance => skip initial hub->first pickup, include last pickup->hub
    if len(optimized_trip) > 2:
        oneway = 0.0
        for i in range(1, len(optimized_trip) - 1):
            dval = calculate_dist(optimized_trip[i][0], optimized_trip[i+1][0])
            if dval is not None:
                oneway += dval
    else:
        oneway = roundtrip

    return {
        "status": "success",
        "hub_id": hub_id,
        "hub_address": hub_address,
        "sequence_of_pickup_id": final_sequence,
        "shortest_distance_oneway": round(oneway, 2),
        "roundtrip_distance": round(roundtrip, 2)
    }

# -------------------------------------------------------------------------
# 4) Worker Loop
# -------------------------------------------------------------------------
def worker_loop():
    """
    Continuously check for PENDING tsp_jobs.
    For each job:
      - Mark RUNNING
      - Perform TSP with process_tsp_job
      - Mark COMPLETE or ERROR
    """
    while True:
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)

        # Find next PENDING job
        cursor.execute("SELECT * FROM tsp_jobs WHERE status='PENDING' ORDER BY created_at LIMIT 1")
        job = cursor.fetchone()

        if not job:
            # No pending jobs => sleep
            conn.close()
            print("No pending jobs. Sleeping 30s.")
            time.sleep(30)
            continue

        job_id = job["id"]
        print(f"Processing TSP job_id={job_id}...")

        # Mark as RUNNING
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("""
            UPDATE tsp_jobs
            SET status='RUNNING', updated_at=%s
            WHERE id=%s
        """, (now_str, job_id))
        conn.commit()

        try:
            # Parse the input JSON
            input_data = json.loads(job["input_json"])

            # Run the TSP
            result_dict = process_tsp_job(input_data)
            output_json = json.dumps(result_dict)

            # Mark COMPLETE
            cursor.execute("""
                UPDATE tsp_jobs
                SET status='COMPLETE', output_json=%s, updated_at=%s
                WHERE id=%s
            """, (output_json, now_str, job_id))
            conn.commit()
            print(f"Job {job_id} completed successfully.")

        except Exception as e:
            # Mark ERROR
            error_msg = traceback.format_exc()
            cursor.execute("""
                UPDATE tsp_jobs
                SET status='ERROR', output_json=%s, updated_at=%s
                WHERE id=%s
            """, (error_msg, now_str, job_id))
            conn.commit()
            print(f"Job {job_id} failed:\n{error_msg}")

        conn.close()

def main():
    print("Starting TSP worker loop...")
    worker_loop()

if __name__ == "__main__":
    main()
