# app.py  – unified Flask front-end
# ─────────────────────────────────────────────────────────────
import json
import uuid
import os
import pandas as pd
import threading
import subprocess
from json import JSONDecodeError
from contextlib import contextmanager
from datetime import datetime

import mysql.connector
import pytz
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename


# ─── Helpers ─────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")


def ist_now() -> str:
    """Current time in IST, formatted for MySQL DATETIME."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


def db():
    """Return a pooled MySQL connection (autocommit ON).

    Reads credentials from environment variables when provided:
      DB_HOST, DB_USER, DB_PASSWORD, DB_NAME

    Falls back to sensible local defaults compatible with csv_route_ui.py.
    """
    host = os.environ.get("DB_HOST", "localhost")
    user = os.environ.get("DB_USER", "root")
    password = os.environ.get("DB_PASSWORD", "rmivuxg")
    database = os.environ.get("DB_NAME", "route_optimization")

    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        pool_name="api_pool",
        pool_size=5,
        autocommit=True,
    )


@contextmanager
def mysql_cursor(dict_rows: bool = False):
    """
    Context-manager that gives a cursor and guarantees the underlying
    connection stays alive until the cursor is closed.
    """
    con = db()
    cur = con.cursor(dictionary=dict_rows)
    try:
        yield cur
    finally:
        cur.close()
        con.close()


# ─── Configuration ───────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# File upload configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Hub mapping file
HUB_MAPPING_FILE = os.path.join(os.path.dirname(__file__), 'hub_mapping.csv')

def load_hub_mapping():
    """Load hub_id to hub_name mapping from hub_mapping.csv if present."""
    try:
        if os.path.exists(HUB_MAPPING_FILE):
            df = pd.read_csv(HUB_MAPPING_FILE)
            if 'hub_id' in df.columns and 'hub_name' in df.columns:
                df['hub_id'] = pd.to_numeric(df['hub_id'], errors='coerce').astype('Int64')
                df = df.dropna(subset=['hub_id', 'hub_name'])
                return [
                    { 'hub_id': int(row['hub_id']), 'hub_name': str(row['hub_name']) }
                    for _, row in df.iterrows()
                ]
    except Exception:
        pass
    return []

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv_structure(file_path):
    """Validate CSV file structure and return validation results."""
    try:
        df = pd.read_csv(file_path)
        
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
        
        # Check data types
        if 'material_quantity' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['material_quantity']):
                validation_result['is_valid'] = False
                validation_result['errors'].append("material_quantity must be numeric")
            elif df['material_quantity'].isna().any():
                validation_result['warnings'].append("Some material_quantity values are missing")
        
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

def validate_weights_against_vehicle_capacity(file_path, hub_id):
    """
    Validate that grouped weights at each location don't exceed available vehicle capacity.
    This prevents optimization failures due to overweight locations.
    
    Args:
        file_path: Path to CSV file
        hub_id: Hub ID to check vehicle configurations
        
    Returns:
        dict with validation results including locations exceeding capacity
    """
    try:
        df = pd.read_csv(file_path)
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'overweight_locations': []
        }
        
        # Check if required columns exist
        if 'material_quantity' not in df.columns or 'state_name' not in df.columns or 'location_pincode' not in df.columns:
            validation_result['warnings'].append("Cannot validate capacity: missing required columns")
            return validation_result
        
        # Clean and format data
        df['state_name'] = df['state_name'].str.upper()
        df['material_quantity'] = pd.to_numeric(df['material_quantity'], errors='coerce')
        df = df.dropna(subset=['material_quantity', 'state_name', 'location_pincode'])
        
        # Create source_state_pincode
        df['source_state_pincode'] = (
            df['state_name'] + '+' + df['location_pincode'].astype(str)
        )
        
        # Group by location and sum weights
        location_weights = (
            df.groupby('source_state_pincode')['material_quantity']
            .sum()
            .reset_index()
        )
        location_weights.columns = ['location', 'total_weight']
        
        # Load vehicle configurations for this hub
        try:
            with mysql_cursor(dict_rows=True) as cur:
                cur.execute("""
                    SELECT vehicle_type, capacity_kg 
                    FROM vehicle_configs 
                    WHERE hub_id = %s AND is_active = TRUE
                    ORDER BY capacity_kg DESC
                """, (hub_id,))
                vehicles = cur.fetchall()
        except Exception as e:
            print(f"Warning: Could not load vehicles from database: {e}")
            vehicles = []
        
        if not vehicles:
            validation_result['warnings'].append(
                f"No vehicles configured for Hub {hub_id}. Please configure vehicles before optimization."
            )
            return validation_result
        
        # Get maximum vehicle capacity
        max_capacity = max(float(v['capacity_kg']) for v in vehicles)
        
        # Check each location's total weight
        overweight_locations = []
        for _, row in location_weights.iterrows():
            location = row['location']
            total_weight = row['total_weight']
            
            if total_weight > max_capacity:
                overweight_locations.append({
                    'location': location,
                    'total_weight': float(total_weight),
                    'max_capacity': float(max_capacity),
                    'excess_weight': float(total_weight - max_capacity)
                })
        
        if overweight_locations:
            validation_result['is_valid'] = False
            validation_result['overweight_locations'] = overweight_locations
            
            # Create detailed error message
            error_details = []
            for loc in overweight_locations[:5]:  # Show first 5
                error_details.append(
                    f"{loc['location']}: {loc['total_weight']:.0f} kg "
                    f"(exceeds max capacity of {loc['max_capacity']:.0f} kg by {loc['excess_weight']:.0f} kg)"
                )
            
            if len(overweight_locations) > 5:
                error_details.append(f"... and {len(overweight_locations) - 5} more locations")
            
            validation_result['errors'].append(
                f"Found {len(overweight_locations)} location(s) with total weight exceeding all available vehicle capacities. " +
                "Please split these shipments into multiple pickups or configure larger vehicles. " +
                "Locations: " + "; ".join(error_details)
            )
        
        # Also check for locations that are close to max capacity (warning)
        near_capacity_locations = []
        capacity_threshold = max_capacity * 0.95  # 95% of max capacity
        for _, row in location_weights.iterrows():
            location = row['location']
            total_weight = row['total_weight']
            
            if capacity_threshold < total_weight <= max_capacity:
                near_capacity_locations.append({
                    'location': location,
                    'total_weight': float(total_weight),
                    'max_capacity': float(max_capacity),
                    'percentage': float((total_weight / max_capacity) * 100)
                })
        
        if near_capacity_locations:
            warning_details = []
            for loc in near_capacity_locations[:3]:  # Show first 3
                warning_details.append(
                    f"{loc['location']}: {loc['total_weight']:.0f} kg "
                    f"({loc['percentage']:.1f}% of max capacity)"
                )
            
            if len(near_capacity_locations) > 3:
                warning_details.append(f"... and {len(near_capacity_locations) - 3} more")
            
            validation_result['warnings'].append(
                f"Found {len(near_capacity_locations)} location(s) near maximum vehicle capacity. " +
                "Locations: " + "; ".join(warning_details)
            )
        
        return validation_result
        
    except Exception as e:
        return {
            'is_valid': True,  # Don't block on validation errors
            'errors': [],
            'warnings': [f"Could not validate vehicle capacity: {str(e)}"],
            'overweight_locations': []
        }


@app.route("/")
def index():
    """Main dashboard page."""
    hub_mapping = load_hub_mapping()
    return render_template('index.html', hub_mapping=hub_mapping)

@app.route("/api")
def api_home():
    """API documentation page."""
    return (
        "Legacy shortest-route:\n"
        "  • POST /api/shortest_route (or /api/shortest_distance)\n"
        "  • GET  /api/shortest_route_status/<job_id>\n\n"
        "Route optimiser:\n"
        "  • POST /api/route_optimise {\"hub_id\": N}\n"
        "  • GET  /api/route_optimise_status/<job_id>\n"
    )


# ═════════════════════════════════════════════════════════════
# ❶ SHORT-ROUTE  (legacy queue)
# ═════════════════════════════════════════════════════════════
@app.route("/api/shortest_route", methods=["POST"])
def shortest_route_queue():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON body provided"}), 400

    request_id = data.get("request_id") or str(uuid.uuid4())
    input_json = json.dumps(data)

    with mysql_cursor() as cur:
        cur.execute(
            """
            INSERT INTO tsp_jobs (request_id, status, input_json, created_at, updated_at)
            VALUES (%s, 'PENDING', %s, %s, %s)
            """,
            (request_id, input_json, ist_now(), ist_now()),
        )
        job_id = cur.lastrowid

    return jsonify({"status": "queued", "job_id": job_id, "request_id": request_id})


@app.route("/api/shortest_distance", methods=["POST"])
def shortest_route_queue_alias():
    return shortest_route_queue()


@app.route("/api/shortest_route_status/<int:job_id>")
def shortest_route_status(job_id: int):
    with mysql_cursor(dict_rows=True) as cur:
        cur.execute("SELECT * FROM tsp_jobs WHERE id=%s", (job_id,))
        job = cur.fetchone()

    if not job:
        return jsonify({"status": "error", "message": "invalid job_id"}), 404

    resp = {
        "job_id": job_id,
        "request_id": job["request_id"],
        "status": job["status"],
    }

    if job["output_json"]:
        try:
            if job["status"] == "COMPLETE":
                resp["result"] = json.loads(job["output_json"])
            elif job["status"] == "ERROR":
                resp["error_details"] = job["output_json"]
        except JSONDecodeError as e:
            # keep the API alive even if someone wrote garbage into output_json
            resp["status"] = "ERROR"
            resp["error_details"] = f"Invalid JSON in output_json: {e}"
            resp["raw_output_json"] = job["output_json"]

    return jsonify(resp)


# ═════════════════════════════════════════════════════════════
# ❷ ROUTE-OPTIMISE  (new queue for the worker)
# ═════════════════════════════════════════════════════════════
@app.route("/api/route_optimise", methods=["POST"])
def route_opt_queue():
    data = request.get_json(silent=True) or {}
    try:
        hub_id = int(data["hub_id"])
    except (KeyError, ValueError):
        return jsonify({"status": "error", "message": "hub_id (int) required"}), 400

    with mysql_cursor(dict_rows=True) as cur:
        # 1) duplicate check
        cur.execute(
            "SELECT id FROM route_opt_jobs WHERE hub_id=%s AND status='PENDING' LIMIT 1",
            (hub_id,),
        )
        dup = cur.fetchone()
        if dup:
            return jsonify({"status": "duplicate_pending", "job_id": dup["id"]}), 200

        # 2) enqueue
        cur.execute(
            """
            INSERT INTO route_opt_jobs (hub_id, status, input_json, created_at)
            VALUES (%s, 'PENDING', %s, %s)
            """,
            (hub_id, json.dumps({"hub_id": hub_id}), ist_now()),
        )
        job_id = cur.lastrowid

    return jsonify({"status": "queued", "job_id": job_id})


@app.route("/api/route_optimise_status/<int:job_id>")
def route_opt_status(job_id: int):
    with mysql_cursor(dict_rows=True) as cur:
        cur.execute("SELECT * FROM route_opt_jobs WHERE id=%s", (job_id,))
        job = cur.fetchone()

    if not job:
        return jsonify({"status": "error", "message": "invalid job_id"}), 404

    # stringify datetimes for JSON
    for col in ("created_at", "started_at", "finished_at"):
        if job.get(col):
            job[col] = job[col].strftime("%Y-%m-%d %H:%M:%S")

    if job.get("output_json"):
        try:
            job["output_json"] = json.loads(job["output_json"])
        except JSONDecodeError as e:
            job["output_json_error"] = f"Invalid JSON in output_json: {e}"

    return jsonify(job)


# ═════════════════════════════════════════════════════════════
# ❸ WEB UI ROUTES
# ═════════════════════════════════════════════════════════════

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
                    # CSV structure is valid - redirect to vehicle configuration
                    flash('CSV file uploaded and validated successfully!', 'success')
                    return redirect(url_for('vehicle_config', filename=filename, hub_id=hub_id_int))
            
            # Validation failed
            if not validation['is_valid']:
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
    
    # Load vehicles from database instead of hardcoded mapping
    try:
        with mysql_cursor(dict_rows=True) as cur:
            cur.execute("""
                SELECT * FROM vehicle_configs 
                WHERE hub_id = %s AND is_active = TRUE 
                ORDER BY vehicle_type
            """, (hub_id,))
            vehicles = cur.fetchall()
    except Exception as e:
        vehicles = []
        flash(f'Error loading vehicle configurations: {e}', 'error')

    return render_template(
        'vehicle_config.html', filename=filename, hub_id=hub_id, vehicles=vehicles
    )

@app.route('/download-sample')
def download_sample():
    """Download preferred sample CSV (from repository file)."""
    sample_path = os.path.join(os.path.dirname(__file__), 'sample_pickup_data.csv')
    if os.path.exists(sample_path):
        return send_file(sample_path, mimetype='text/csv', as_attachment=True,
                         download_name='sample_pickup_data.csv')
    # Fallback small inline sample matching schema
    sample_data = (
        "pickup_number,material_quantity,city_name,state_name,location_pincode\n"
        "P001,150,Chennai,Tamil Nadu,600001\n"
        "P002,200,Chennai,Tamil Nadu,600002\n"
    )
    from io import BytesIO
    return send_file(BytesIO(sample_data.encode('utf-8')), mimetype='text/csv', as_attachment=True,
                     download_name='sample_pickup_data.csv')

@app.route('/optimize', methods=['POST'])
def start_optimization():
    """Start route optimization process."""
    data = request.get_json()
    filename = data.get('filename')
    hub_id = data.get('hub_id')
    
    if not filename or not hub_id:
        return jsonify({'status': 'error', 'message': 'Missing filename or hub_id'}), 400
    
    # Validate file path
    input_file = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(input_file):
        return jsonify({'status': 'error', 'message': 'Input file not found'}), 400
    
    # Validate weights against vehicle capacity BEFORE starting optimization
    try:
        capacity_validation = validate_weights_against_vehicle_capacity(input_file, hub_id)
        
        if not capacity_validation['is_valid']:
            # Build detailed error message with overweight locations
            error_message = "Cannot start optimization: Some locations exceed available vehicle capacity.\n\n"
            
            if capacity_validation.get('overweight_locations'):
                error_message += "Overweight Locations:\n"
                for loc in capacity_validation['overweight_locations']:
                    error_message += f"• {loc['location']}: {loc['total_weight']:.0f} kg "
                    error_message += f"(exceeds max capacity of {loc['max_capacity']:.0f} kg)\n"
                
                error_message += f"\nPlease either:\n"
                error_message += f"1. Add a vehicle with capacity > {max(loc['total_weight'] for loc in capacity_validation['overweight_locations']):.0f} kg for this hub\n"
                error_message += f"2. Split the shipments at these locations into multiple pickup entries"
            
            return jsonify({
                'status': 'error', 
                'message': error_message,
                'overweight_locations': capacity_validation.get('overweight_locations', [])
            }), 400
    
    except Exception as e:
        # Don't block optimization if validation itself fails
        print(f"Warning: Capacity validation failed: {e}")
    
    # Generate unique job ID (shortened to 8 characters)
    job_id = str(uuid.uuid4())[:8]
    
    # Create job record
    try:
        with mysql_cursor() as cur:
            output_file = os.path.join(RESULTS_FOLDER, f"optimized_{job_id}.csv")
            
            cur.execute("""
                INSERT INTO optimization_jobs 
                (job_id, hub_id, input_file, output_file, status)
                VALUES (%s, %s, %s, %s, 'PENDING')
            """, (job_id, hub_id, input_file, output_file))
        
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
        with mysql_cursor(dict_rows=True) as cur:
            cur.execute("""
                SELECT * FROM optimization_jobs WHERE job_id = %s
            """, (job_id,))
            
            job = cur.fetchone()
        
        if not job:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404
        
        # Convert datetime objects to strings
        for field in ['created_at', 'started_at', 'finished_at']:
            if job.get(field):
                job[field] = job[field].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(job)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/vehicles/<int:hub_id>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def manage_vehicles(hub_id):
    """API endpoint for vehicle configuration management."""
    try:
        if request.method == 'GET':
            # Get all vehicles for hub
            with mysql_cursor(dict_rows=True) as cur:
                cur.execute("""
                    SELECT * FROM vehicle_configs 
                    WHERE hub_id = %s AND is_active = TRUE 
                    ORDER BY vehicle_type
                """, (hub_id,))
                vehicles = cur.fetchall()
            return jsonify({'vehicles': vehicles})
        
        elif request.method == 'POST':
            # Add new vehicle
            data = request.get_json()
            with mysql_cursor() as cur:
                cur.execute("""
                    INSERT INTO vehicle_configs 
                    (hub_id, vehicle_type, capacity_kg, max_locations, cost_per_km, max_distance_km)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (hub_id, data['vehicle_type'], data['capacity_kg'], 
                      data['max_locations'], data['cost_per_km'], data['max_distance_km']))
            return jsonify({'status': 'success', 'message': 'Vehicle added successfully'})
        
        elif request.method == 'PUT':
            # Update existing vehicle
            data = request.get_json()
            vehicle_id = data['id']
            with mysql_cursor() as cur:
                cur.execute("""
                    UPDATE vehicle_configs SET
                    vehicle_type = %s, capacity_kg = %s, max_locations = %s,
                    cost_per_km = %s, max_distance_km = %s
                    WHERE id = %s AND hub_id = %s
                """, (data['vehicle_type'], data['capacity_kg'], data['max_locations'],
                      data['cost_per_km'], data['max_distance_km'], vehicle_id, hub_id))
            return jsonify({'status': 'success', 'message': 'Vehicle updated successfully'})
        
        elif request.method == 'DELETE':
            # Soft delete vehicle (set is_active = FALSE)
            vehicle_id = request.args.get('id')
            with mysql_cursor() as cur:
                cur.execute("""
                    UPDATE vehicle_configs SET is_active = FALSE 
                    WHERE id = %s AND hub_id = %s
                """, (vehicle_id, hub_id))
            return jsonify({'status': 'success', 'message': 'Vehicle removed successfully'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download/<job_id>')
def download_results(job_id):
    """Download optimization results."""
    try:
        with mysql_cursor(dict_rows=True) as cur:
            cur.execute("""
                SELECT output_file FROM optimization_jobs WHERE job_id = %s
            """, (job_id,))
            
            job = cur.fetchone()
        
        if not job:
            flash('Job not found', 'error')
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
    """Return dashboard stats and recent optimization jobs."""
    try:
        with mysql_cursor(dict_rows=True) as cur:
            # Aggregate counts
            cur.execute(
                """
                SELECT 
                    COUNT(*) AS total,
                    SUM(CASE WHEN status='COMPLETE' THEN 1 ELSE 0 END) AS completed,
                    SUM(CASE WHEN status='RUNNING' THEN 1 ELSE 0 END) AS running,
                    SUM(CASE WHEN status='ERROR' THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) AS pending
                FROM optimization_jobs
                """
            )
            stats = cur.fetchone() or {}

            # Recent jobs
            cur.execute(
                """
                SELECT job_id, hub_id, status, status_message, created_at, finished_at
                FROM optimization_jobs
                ORDER BY created_at DESC
                LIMIT 10
                """
            )
            recent = cur.fetchall() or []

        # Serialize datetimes
        def fmt(ts):
            try:
                return ts.strftime('%Y-%m-%d %H:%M:%S') if ts else None
            except Exception:
                return str(ts) if ts is not None else None

        for j in recent:
            j['created_at'] = fmt(j.get('created_at'))
            j['finished_at'] = fmt(j.get('finished_at'))

        return jsonify({ 'stats': stats, 'recent_jobs': recent })
    except Exception as e:
        return jsonify({ 'error': str(e) }), 500

@app.route('/api/job-results/<job_id>')
def job_results(job_id):
    """Get detailed results for a completed optimization job."""
    try:
        with mysql_cursor(dict_rows=True) as cur:
            cur.execute("""
                SELECT * FROM optimization_jobs WHERE job_id = %s
            """, (job_id,))
            
            job = cur.fetchone()
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Check if job is completed or if we have results file (for ERROR jobs that actually completed)
        if job['status'] != 'COMPLETE':
            # Check if results file exists - sometimes jobs complete but status isn't updated properly
            output_file = job['output_file']
            if not os.path.exists(output_file):
                return jsonify({'error': 'Job not completed yet'}), 400
            # If results file exists, treat as completed for viewing purposes
        
        # Read the results CSV file
        output_file = job['output_file']
        if not os.path.exists(output_file):
            return jsonify({'error': 'Results file not found'}), 404
        
        try:
            df = pd.read_csv(output_file)
            
            # Calculate summary statistics
            total_trips = int(df['trip_number'].nunique())
            total_distance = float(df['total_distance_km'].sum())
            total_cost = float(df['cost_inr'].sum())
            total_weight = float(df['total_weight_kg'].sum())
            
            # Get algorithm used (should be same for all rows)
            algorithm_used = str(df['algorithm_used'].iloc[0]) if len(df) > 0 else 'Unknown'
            
            # Calculate cost per kg
            cost_per_kg = float(total_cost / total_weight) if total_weight > 0 else 0.0
            
            # Calculate average distance per trip
            avg_distance_per_trip = float(total_distance / total_trips) if total_trips > 0 else 0.0
            
            # Group by trip to get trip details
            trips = []
            for trip_num in sorted(df['trip_number'].unique()):
                trip_data = df[df['trip_number'] == trip_num]
                
                # Get trip summary (from first row of each trip)
                first_row = trip_data.iloc[0]
                trip_summary = {
                    'trip_number': int(trip_num),
                    'vehicle_type': str(first_row['vehicle_type']),
                    'total_distance': float(first_row['total_distance_km']),
                    'total_weight': float(first_row['total_weight_kg']),
                    'cost': float(first_row['cost_inr']),
                    'cost_per_kg': float(first_row['cost_per_kg']),
                    'stops': []
                }
                
                # Get all stops for this trip
                for _, stop in trip_data.iterrows():
                    trip_summary['stops'].append({
                        'sequence': int(stop['stop_sequence']),
                        'location': str(stop['location']),
                        'pickup_numbers': str(stop['pickup_numbers']),
                        'weight_kg': float(stop['weight_kg'])
                    })
                
                trips.append(trip_summary)
            
            # Calculate total locations and pickups
            total_locations = int(df['location'].nunique())
            total_pickups = int(df['pickup_numbers'].str.count(',').sum() + len(df))  # Count commas + 1 for each row
            
            result = {
                'job_id': job_id,
                'total_trips': total_trips,
                'total_distance': total_distance,
                'total_cost': total_cost,
                'total_weight': total_weight,
                'cost_per_kg': cost_per_kg,
                'avg_distance_per_trip': avg_distance_per_trip,
                'algorithm_used': algorithm_used,
                'total_locations': total_locations,
                'total_pickups': total_pickups,
                'trips': trips
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Error reading results file: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Database initialization
def create_required_tables():
    """Create required database tables if they don't exist."""
    try:
        with mysql_cursor() as cur:
            # Create vehicle_configs table
            cur.execute("""
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
            
            # Create route_opt_jobs table (used by /api/route_optimise)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS route_opt_jobs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    hub_id INT NOT NULL,
                    status ENUM('PENDING','RUNNING','COMPLETE','ERROR') DEFAULT 'PENDING',
                    input_json TEXT,
                    output_json TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP NULL,
                    finished_at TIMESTAMP NULL,
                    INDEX idx_status (status),
                    INDEX idx_hub_id (hub_id),
                    INDEX idx_created_at (created_at)
                )
            """)

            # Create optimization_jobs table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS optimization_jobs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    job_id VARCHAR(100) UNIQUE NOT NULL,
                    hub_id INT NOT NULL,
                    input_file VARCHAR(500),
                    output_file VARCHAR(500),
                    status ENUM('PENDING', 'RUNNING', 'COMPLETE', 'ERROR') DEFAULT 'PENDING',
                    status_message VARCHAR(500) DEFAULT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP NULL,
                    finished_at TIMESTAMP NULL,
                    INDEX idx_status (status),
                    INDEX idx_hub_id (hub_id)
                )
            """)
        
        print("Database tables initialized successfully!")
        return True
    except Exception as e:
        print(f"Error creating database tables: {e}")
        return False

# Background optimization process
def run_optimization(job_id, input_file, output_file, hub_id):
    """Run optimization in background thread."""
    try:
        # Update job status to running
        with mysql_cursor() as cur:
            cur.execute("""
                UPDATE optimization_jobs 
                SET status = 'RUNNING', started_at = NOW(), status_message = 'Starting route optimization...'
                WHERE job_id = %s
            """, (job_id,))
        
        # Run the CSV route optimizer
        cmd = [
            'python', 'csv_route_optimizer.py',
            input_file, output_file, str(hub_id), '--job-id', job_id
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            # Update job status to complete
            with mysql_cursor() as cur:
                cur.execute("""
                    UPDATE optimization_jobs 
                    SET status = 'COMPLETE', finished_at = NOW(), status_message = 'Route optimization completed successfully'
                    WHERE job_id = %s
                """, (job_id,))
        else:
            # Update job status to error
            with mysql_cursor() as cur:
                cur.execute("""
                    UPDATE optimization_jobs 
                    SET status = 'ERROR', finished_at = NOW(), error_message = %s, status_message = 'Route optimization failed'
                    WHERE job_id = %s
                """, (result.stderr, job_id))
            
    except subprocess.TimeoutExpired:
        # Handle timeout
        with mysql_cursor() as cur:
            cur.execute("""
                UPDATE optimization_jobs 
                SET status = 'ERROR', finished_at = NOW(), error_message = 'Optimization timed out', status_message = 'Route optimization timed out'
                WHERE job_id = %s
            """, (job_id,))
        
    except Exception as e:
        # Handle other errors
        with mysql_cursor() as cur:
            cur.execute("""
                UPDATE optimization_jobs 
                SET status = 'ERROR', finished_at = NOW(), error_message = %s, status_message = 'Route optimization failed'
                WHERE job_id = %s
            """, (str(e), job_id))


# ─── WSGI entry-point (PythonAnywhere picks up `application`) ─────────
application = app

if __name__ == "__main__":
    # Initialize database tables
    print("Initializing database tables...")
    create_required_tables()
    
    # Only used when you run `python app.py` locally
    app.run(debug=True, port=8000)
