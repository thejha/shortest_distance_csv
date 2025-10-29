#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV Route Optimization Web UI
=============================

A comprehensive web interface for CSV-based route optimization with:
- CSV file upload and validation
- Vehicle configuration management
- Real-time optimization progress
- Results visualization and download

Usage:
    python csv_route_ui.py

Features:
- Modern responsive design
- CSV validation with detailed error reporting
- Vehicle fleet management (add/edit/remove vehicles)
- Real-time optimization status updates
- Download optimized routes as CSV
"""

import os
import json
import uuid
import pandas as pd
import mysql.connector
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import subprocess
import threading
import time

# ------------------------------------------------
# Configuration
# ------------------------------------------------
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# File upload configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# MySQL database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "rmivuxg",
}

# Optional hub mapping file (project root): hub_mapping.csv with columns: hub_id,hub_name
HUB_MAPPING_FILE = os.path.join(os.path.dirname(__file__), 'hub_mapping.csv')

def load_hub_mapping():
    """Load hub_id to hub_name mapping from hub_mapping.csv if present.
    Returns list of dicts: [{hub_id:int, hub_name:str}]"""
    try:
        if os.path.exists(HUB_MAPPING_FILE):
            df = pd.read_csv(HUB_MAPPING_FILE)
            if 'hub_id' in df.columns and 'hub_name' in df.columns:
                # Coerce hub_id to int if possible
                df['hub_id'] = pd.to_numeric(df['hub_id'], errors='coerce').astype('Int64')
                df = df.dropna(subset=['hub_id', 'hub_name'])
                return [
                    { 'hub_id': int(row['hub_id']), 'hub_name': str(row['hub_name']) }
                    for _, row in df.iterrows()
                ]
    except Exception:
        pass
    # Fallback: empty mapping
    return []

# ------------------------------------------------
# Database helpers
# ------------------------------------------------
def get_mysql_connection():
    """Create a connection to the MySQL database."""
    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database="route_optimization"
    )

def create_vehicle_config_table():
    """Create vehicle configuration table if it doesn't exist."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
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
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error creating vehicle config table: {e}")
        return False

def create_optimization_jobs_table():
    """Create optimization jobs table if it doesn't exist."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_jobs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                job_id VARCHAR(100) UNIQUE NOT NULL,
                hub_id INT NOT NULL,
                input_file VARCHAR(500),
                output_file VARCHAR(500),
                status ENUM('PENDING', 'RUNNING', 'COMPLETED', 'ERROR') DEFAULT 'PENDING',
                status_message VARCHAR(500) DEFAULT NULL,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP NULL,
                finished_at TIMESTAMP NULL,
                INDEX idx_status (status),
                INDEX idx_hub_id (hub_id)
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error creating optimization jobs table: {e}")
        return False

# ------------------------------------------------
# Utility functions
# ------------------------------------------------
def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv_structure(file_path):
    """Validate CSV file structure and return validation results."""
    try:
        df = pd.read_csv(file_path)
        
        # Required columns (hub_id now selected via UI, not in CSV)
        required_columns = [
            'pickup_number', 'material_quantity', 'city_name', 'state_name', 'location_pincode'
        ]
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'row_count': len(df),
            'columns': list(df.columns)
        }
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check data types and values (light checks per schema)
        if 'material_quantity' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['material_quantity']):
                validation_result['is_valid'] = False
                validation_result['errors'].append("material_quantity must be numeric")
            elif df['material_quantity'].isna().any():
                validation_result['warnings'].append("Some material_quantity values are missing")

        # hub_id is chosen via UI dropdown; no CSV validation needed for it
        
        # Check for empty rows
        if df.isnull().all(axis=1).any():
            validation_result['warnings'].append("Some rows appear to be completely empty")
        
        return validation_result
        
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Error reading CSV file: {str(e)}"],
            'warnings': [],
            'row_count': 0,
            'columns': []
        }

# ------------------------------------------------
# Route handlers
# ------------------------------------------------
@app.route('/')
def index():
    """Main dashboard page."""
    hub_mapping = load_hub_mapping()
    return render_template('index.html', hub_mapping=hub_mapping)

@app.route('/hub-mapping')
def view_hub_mapping():
    """Hub mapping page with detailed table."""
    raw = load_hub_mapping()
    # Enrich and sort
    enriched = []
    for item in raw:
        hub_name = item.get('hub_name', '')
        pincode = ''
        if '(' in hub_name and ')' in hub_name:
            try:
                pincode = hub_name.split('(')[1].split(')')[0]
            except Exception:
                pincode = ''
        enriched.append({
            'hub_id': item.get('hub_id'),
            'hub_name': hub_name,
            'pincode': pincode,
        })
    enriched.sort(key=lambda x: x['hub_id'] if x['hub_id'] is not None else 0)
    return render_template('hub_mapping.html', hub_mapping=enriched)

@app.route('/upload', methods=['GET', 'POST'])
def upload_csv():
    """Handle CSV file upload and validation."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            # Validate CSV structure
            validation = validate_csv_structure(file_path)
            
            if validation['is_valid']:
                # Require hub_id from UI dropdown
                selected_hub = request.form.get('hub_id') or request.args.get('hub_id')
                try:
                    hub_id_int = int(selected_hub) if selected_hub is not None else None
                except ValueError:
                    hub_id_int = None

                if hub_id_int is None:
                    validation['is_valid'] = False
                    validation['errors'].append('Please select a Hub from the dropdown before uploading.')
                else:
                    flash('CSV file uploaded and validated successfully!', 'success')
                    return render_template('vehicle_config.html', 
                                           filename=filename, 
                                           hub_id=hub_id_int,
                                           validation=validation)
            else:
                flash('CSV validation failed', 'error')
                return render_template('upload_result.html', 
                                    filename=filename, 
                                    validation=validation)
        else:
            flash('Invalid file type. Please upload a CSV file.', 'error')
            return redirect(request.url)
    
    hub_mapping = load_hub_mapping()
    return render_template('upload.html', hub_mapping=hub_mapping)

@app.route('/vehicle-config/<filename>')
def vehicle_config(filename):
    """Vehicle configuration page."""
    hub_id = request.args.get('hub_id', type=int)
    if not hub_id:
        flash('Hub ID is required', 'error')
        return redirect(url_for('upload_csv'))
    
    # Get existing vehicle configurations for this hub
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM vehicle_configs 
            WHERE hub_id = %s AND is_active = TRUE 
            ORDER BY vehicle_type
        """, (hub_id,))
        vehicles = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        vehicles = []
        flash(f'Error loading vehicle configurations: {e}', 'error')
    
    return render_template('vehicle_config.html', 
                         filename=filename, 
                         hub_id=hub_id, 
                         vehicles=vehicles)

@app.route('/api/vehicles/<int:hub_id>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def manage_vehicles(hub_id):
    """API endpoint for vehicle configuration management."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        
        if request.method == 'GET':
            # Get all vehicles for hub
            cursor.execute("""
                SELECT * FROM vehicle_configs 
                WHERE hub_id = %s AND is_active = TRUE 
                ORDER BY vehicle_type
            """, (hub_id,))
            vehicles = cursor.fetchall()
            return jsonify({'vehicles': vehicles})
        
        elif request.method == 'POST':
            # Add new vehicle
            data = request.get_json()
            cursor.execute("""
                INSERT INTO vehicle_configs 
                (hub_id, vehicle_type, capacity_kg, max_locations, cost_per_km, max_distance_km)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (hub_id, data['vehicle_type'], data['capacity_kg'], 
                  data['max_locations'], data['cost_per_km'], data['max_distance_km']))
            conn.commit()
            return jsonify({'status': 'success', 'message': 'Vehicle added successfully'})
        
        elif request.method == 'PUT':
            # Update existing vehicle
            data = request.get_json()
            vehicle_id = data['id']
            cursor.execute("""
                UPDATE vehicle_configs SET
                vehicle_type = %s, capacity_kg = %s, max_locations = %s,
                cost_per_km = %s, max_distance_km = %s
                WHERE id = %s AND hub_id = %s
            """, (data['vehicle_type'], data['capacity_kg'], data['max_locations'],
                  data['cost_per_km'], data['max_distance_km'], vehicle_id, hub_id))
            conn.commit()
            return jsonify({'status': 'success', 'message': 'Vehicle updated successfully'})
        
        elif request.method == 'DELETE':
            # Soft delete vehicle (set is_active = FALSE)
            vehicle_id = request.args.get('id')
            cursor.execute("""
                UPDATE vehicle_configs SET is_active = FALSE 
                WHERE id = %s AND hub_id = %s
            """, (vehicle_id, hub_id))
            conn.commit()
            return jsonify({'status': 'success', 'message': 'Vehicle removed successfully'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/optimize', methods=['POST'])
def start_optimization():
    """Start route optimization process."""
    data = request.get_json()
    filename = data.get('filename')
    hub_id = data.get('hub_id')
    
    if not filename or not hub_id:
        return jsonify({'status': 'error', 'message': 'Missing filename or hub_id'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job record
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        
        input_file = os.path.join(UPLOAD_FOLDER, filename)
        output_file = os.path.join(RESULTS_FOLDER, f"optimized_{job_id}.csv")
        
        cursor.execute("""
            INSERT INTO optimization_jobs 
            (job_id, hub_id, input_file, output_file, status)
            VALUES (%s, %s, %s, %s, 'PENDING')
        """, (job_id, hub_id, input_file, output_file))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Start optimization in background thread
        thread = threading.Thread(target=run_optimization, args=(job_id, input_file, output_file, hub_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'job_id': job_id})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/optimization-status/<job_id>')
def optimization_status(job_id):
    """Get optimization job status."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT * FROM optimization_jobs WHERE job_id = %s
        """, (job_id,))
        
        job = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not job:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404
        
        # Convert datetime objects to strings
        for field in ['created_at', 'started_at', 'finished_at']:
            if job[field]:
                job[field] = job[field].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(job)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download/<job_id>')
def download_results(job_id):
    """Download optimization results."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT output_file FROM optimization_jobs WHERE job_id = %s AND status = 'COMPLETED'
        """, (job_id,))
        
        job = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not job:
            flash('Results not available', 'error')
            return redirect(url_for('index'))
        
        output_file = job['output_file']
        if os.path.exists(output_file):
            return send_file(output_file, as_attachment=True, 
                           download_name=f'optimized_routes_{job_id}.csv')
        else:
            flash('Results file not found', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Error downloading results: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/api/dashboard')
def dashboard_data():
    """Get dashboard statistics and recent jobs."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get job counts by status
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'RUNNING' THEN 1 ELSE 0 END) as running,
                SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) as pending
            FROM optimization_jobs
        """)
        stats = cursor.fetchone() or {}
        
        # Get recent jobs (last 10)
        cursor.execute("""
            SELECT job_id, hub_id, status, status_message, created_at, finished_at
            FROM optimization_jobs 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        recent_jobs = cursor.fetchall() or []

        # Normalize/serialize datetime fields to strings for JSON
        def serialize_dt(dt):
            try:
                return dt.strftime('%Y-%m-%d %H:%M:%S') if dt else None
            except Exception:
                return str(dt) if dt is not None else None

        for j in recent_jobs:
            if 'created_at' in j:
                j['created_at'] = serialize_dt(j.get('created_at'))
            if 'finished_at' in j:
                j['finished_at'] = serialize_dt(j.get('finished_at'))
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'stats': stats,
            'recent_jobs': recent_jobs
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-sample')
def download_sample():
    """Download preferred sample CSV (from repository file)."""
    sample_path = os.path.join(os.path.dirname(__file__), 'sample_pickup_data.csv')
    if os.path.exists(sample_path):
        return send_file(sample_path, mimetype='text/csv', as_attachment=True,
                         download_name='sample_pickup_data.csv')
    # Fallback small inline sample matching schema
    sample_data = (
        "pickup_number,material_quantity,city_name,state_name,location_pincode,hub_id\n"
        "P001,150,Chennai,Tamil Nadu,600001,8\n"
        "P002,200,Chennai,Tamil Nadu,600002,8\n"
    )
    from io import BytesIO
    return send_file(BytesIO(sample_data.encode('utf-8')), mimetype='text/csv', as_attachment=True,
                     download_name='sample_pickup_data.csv')

# ------------------------------------------------
# Background optimization process
# ------------------------------------------------
def run_optimization(job_id, input_file, output_file, hub_id):
    """Run optimization in background thread."""
    try:
        # Update job status to running
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE optimization_jobs 
            SET status = 'RUNNING', started_at = NOW(), status_message = 'Starting route optimization...'
            WHERE job_id = %s
        """, (job_id,))
        conn.commit()
        cursor.close()
        conn.close()
        
        # Run the CSV route optimizer
        cmd = [
            'python', 'csv_route_optimizer.py',
            input_file, output_file, str(hub_id), '--job-id', job_id
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            # Update job status to complete
            conn = get_mysql_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE optimization_jobs 
                SET status = 'COMPLETED', finished_at = NOW(), status_message = 'Route optimization completed successfully'
                WHERE job_id = %s
            """, (job_id,))
            conn.commit()
            cursor.close()
            conn.close()
        else:
            # Update job status to error
            conn = get_mysql_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE optimization_jobs 
                SET status = 'ERROR', finished_at = NOW(), error_message = %s, status_message = 'Route optimization failed'
                WHERE job_id = %s
            """, (result.stderr, job_id))
            conn.commit()
            cursor.close()
            conn.close()
            
    except subprocess.TimeoutExpired:
        # Handle timeout
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE optimization_jobs 
            SET status = 'ERROR', finished_at = NOW(), error_message = 'Optimization timed out', status_message = 'Route optimization timed out'
            WHERE job_id = %s
        """, (job_id,))
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        # Handle other errors
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE optimization_jobs 
            SET status = 'ERROR', finished_at = NOW(), error_message = %s, status_message = 'Route optimization failed'
            WHERE job_id = %s
        """, (str(e), job_id))
        conn.commit()
        cursor.close()
        conn.close()

@app.route('/api/job-results/<job_id>')
def get_job_results(job_id):
    """Get detailed results for a completed optimization job."""
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get job details
        cursor.execute("""
            SELECT * FROM optimization_jobs WHERE job_id = %s AND status = 'COMPLETED'
        """, (job_id,))
        
        job = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not job:
            return jsonify({'error': 'Job not found or not completed'}), 404
        
        # Read the output CSV file to extract results
        output_file = job['output_file']
        if not os.path.exists(output_file):
            return jsonify({'error': 'Results file not found'}), 404
        
        # Parse CSV results
        df = pd.read_csv(output_file)
        
        # Calculate summary statistics
        total_trips = df['trip_number'].nunique()
        total_distance = df['total_distance_km'].sum() if 'total_distance_km' in df.columns else 0
        total_cost = df['cost_inr'].sum() if 'cost_inr' in df.columns else 0
        total_weight = df['weight_kg'].sum()
        
        # Get unique locations and pickups
        total_locations = df['location'].nunique()
        total_pickups = df['pickup_numbers'].str.count(',').sum() + len(df)  # Count commas + 1 for each row
        
        # Calculate cost per kg
        cost_per_kg = total_cost / total_weight if total_weight > 0 else 0
        
        # Calculate average distance per trip
        avg_distance_per_trip = total_distance / total_trips if total_trips > 0 else 0
        
        # Get algorithm used
        algorithm_used = df['algorithm_used'].iloc[0] if 'algorithm_used' in df.columns else 'Unknown'
        
        # Group trips and format trip details
        trips = []
        for trip_num in sorted(df['trip_number'].unique()):
            trip_df = df[df['trip_number'] == trip_num]
            
            # Get trip statistics (from first row which has the totals)
            first_row = trip_df.iloc[0]
            trip_distance = first_row.get('total_distance_km', 0)
            trip_cost = first_row.get('cost_inr', 0)
            trip_cost_per_kg = first_row.get('cost_per_kg', 0)
            vehicle_type = first_row.get('vehicle_type', 'Unknown')
            
            # Build route string
            locations = trip_df['location'].tolist()
            route = 'Hub -> ' + ' -> '.join(locations) + ' -> Hub'
            
            trips.append({
                'trip_number': int(trip_num),
                'total_distance': float(trip_distance),
                'cost': float(trip_cost),
                'cost_per_kg': float(trip_cost_per_kg),
                'vehicle_type': vehicle_type,
                'route': route,
                'stops': [{'location': loc} for loc in locations]
            })
        
        results = {
            'total_locations': int(total_locations),
            'total_pickups': int(total_pickups),
            'total_cost': float(total_cost),
            'cost_per_kg': float(cost_per_kg),
            'total_trips': int(total_trips),
            'total_distance': float(total_distance),
            'avg_distance_per_trip': float(avg_distance_per_trip),
            'algorithm_used': algorithm_used,
            'trips': trips
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------
# Initialize database tables
# ------------------------------------------------
def initialize_database():
    """Initialize required database tables."""
    create_vehicle_config_table()
    create_optimization_jobs_table()

# ------------------------------------------------
# Main execution
# ------------------------------------------------
if __name__ == '__main__':
    print("Initializing CSV Route Optimization Web UI...")
    initialize_database()
    print("Database tables initialized successfully!")
    print("Starting web server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
