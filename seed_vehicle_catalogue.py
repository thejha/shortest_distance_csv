#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-time seeding script to load the vehicle catalogue into the database
for all hubs as individual rows in `vehicle_configs`.

Reads DB connection from environment variables (same as app.py):
  - DB_HOST (default: localhost)
  - DB_USER (default: root)
  - DB_PASSWORD (default: rmivuxg)
  - DB_NAME (default: route_optimization)

Usage:
  Windows PowerShell:
    $env:DB_HOST="localhost"; $env:DB_USER="root"; $env:DB_PASSWORD="rmivuxg"; $env:DB_NAME="route_optimization"
    python seed_vehicle_catalogue.py
"""

import os
import sys
import mysql.connector


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


def get_db_connection():
    host = os.environ.get("DB_HOST", "localhost")
    user = os.environ.get("DB_USER", "root")
    password = os.environ.get("DB_PASSWORD", "rmivuxg")
    database = os.environ.get("DB_NAME", "route_optimization")
    return mysql.connector.connect(host=host, user=user, password=password, database=database)


def ensure_tables():
    con = get_db_connection()
    cur = con.cursor()
    # Ensure table exists
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_configs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            hub_id INT NOT NULL,
            vehicle_type VARCHAR(100) NOT NULL,
            capacity_kg DECIMAL(10,2) NOT NULL,
            max_locations INT NOT NULL,
            cost_per_km DECIMAL(8,2) NOT NULL,
            max_distance_km DECIMAL(8,2) NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_hub_id (hub_id),
            INDEX idx_active (is_active)
        )
        """
    )
    con.commit()

    # Inspect columns and add any missing (for older installs)
    cur.execute("DESCRIBE vehicle_configs")
    existing = {row[0] for row in cur.fetchall()}
    # Drop columns not expected by our schema (fix legacy installs)
    if "fuel_efficiency" in existing:
        cur.execute("ALTER TABLE vehicle_configs DROP COLUMN fuel_efficiency")
        con.commit()
        # refresh columns after drop
        cur.execute("DESCRIBE vehicle_configs")
        existing = {row[0] for row in cur.fetchall()}
    migrations = []
    if "capacity_kg" not in existing:
        migrations.append("ADD COLUMN capacity_kg DECIMAL(10,2) NOT NULL DEFAULT 0 AFTER vehicle_type")
    if "max_locations" not in existing:
        migrations.append("ADD COLUMN max_locations INT NOT NULL DEFAULT 1 AFTER capacity_kg")
    if "cost_per_km" not in existing:
        migrations.append("ADD COLUMN cost_per_km DECIMAL(8,2) NOT NULL DEFAULT 0 AFTER max_locations")
    if "max_distance_km" not in existing:
        migrations.append("ADD COLUMN max_distance_km DECIMAL(8,2) NOT NULL DEFAULT 0 AFTER cost_per_km")
    if "is_active" not in existing:
        migrations.append("ADD COLUMN is_active BOOLEAN DEFAULT TRUE AFTER max_distance_km")

    for mig in migrations:
        cur.execute(f"ALTER TABLE vehicle_configs {mig}")
    if migrations:
        con.commit()

    cur.close()
    con.close()


def upsert_vehicle(cur, hub_id: int, vehicle_type: str, capacity: float, max_locations: int,
                   cost_per_km: float, range_km: float) -> None:
    # Try update first; if 0 rows affected, insert
    cur.execute(
        """
        UPDATE vehicle_configs SET
            capacity_kg = %s,
            max_locations = %s,
            cost_per_km = %s,
            max_distance_km = %s,
            is_active = TRUE
        WHERE hub_id = %s AND vehicle_type = %s
        """,
        (capacity, max_locations, cost_per_km, range_km, hub_id, vehicle_type)
    )
    if cur.rowcount == 0:
        cur.execute(
            """
            INSERT INTO vehicle_configs
                (hub_id, vehicle_type, capacity_kg, max_locations, cost_per_km, max_distance_km, is_active)
            VALUES
                (%s, %s, %s, %s, %s, %s, TRUE)
            """,
            (hub_id, vehicle_type, capacity, max_locations, cost_per_km, range_km)
        )


def seed_catalogue() -> int:
    ensure_tables()
    con = get_db_connection()
    cur = con.cursor()
    inserted_or_updated = 0
    try:
        for vtype, spec in VEHICLES.items():
            capacity = float(spec["capacity"])
            cost_per_km = float(spec["cost_per_km"])
            max_locations = int(spec["max_locations"])
            range_km = float(spec["range_km"])
            hubs = list(spec.get("hub_id", []) or [])
            for hub_id in hubs:
                upsert_vehicle(cur, int(hub_id), vtype, capacity, max_locations, cost_per_km, range_km)
                inserted_or_updated += 1
        con.commit()
        return inserted_or_updated
    finally:
        cur.close()
        con.close()


def main():
    try:
        total = seed_catalogue()
        print(f"Seed complete. Rows inserted/updated: {total}")
    except Exception as e:
        print(f"Error during seeding: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


