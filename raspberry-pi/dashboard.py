"""
Cyanobacteria Detection System - Web Dashboard
Flask server that reads sensor CSV data and serves a real-time monitoring dashboard.
"""

import os
import csv
import json
import threading
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# ── CONFIGURATION ──
CSV_PATH = os.environ.get(
    "SENSOR_CSV_PATH",
    os.path.join(PROJECT_ROOT, "data", "sensor_data.csv")
)
ARCHIVE_DIR = os.path.expanduser("~/sensor_data_archive")
DATA_RETENTION_DAYS = 7

# ── PI SENSOR PULL CONFIG ──
# Set to the Pi's sensor_logger API URL to pull readings automatically.
# Set to None or empty string to disable.
PI_SENSOR_URL = os.environ.get("PI_SENSOR_URL", "http://192.168.68.111:8000")
PI_PULL_INTERVAL = 10  # seconds between pulls

# Time range mappings (range key -> timedelta)
RANGE_MAP = {
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}


def read_csv_data(time_range="6h"):
    """Read sensor data from CSV, filtered by time range."""
    data_path = Path(CSV_PATH)
    if not data_path.exists():
        return None

    cutoff = datetime.now() - RANGE_MAP.get(time_range, timedelta(hours=6))
    rows = []

    try:
        with open(data_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    if ts >= cutoff:
                        rows.append({
                            "timestamp": row["timestamp"],
                            "temperature": float(row["temperature"]),
                            "turbidity": float(row["turbidity"]),
                            "ph": float(row["ph"]),
                            "conductivity": float(row["conductivity"]),
                        })
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    return rows


def compute_stats(rows):
    """Compute summary statistics from data rows."""
    if not rows:
        return {
            "total_readings": 0,
            "uptime_hours": 0,
            "min": None,
            "max": None,
            "avg": None,
        }

    sensors = ["temperature", "turbidity", "ph", "conductivity"]
    mins = {s: min(r[s] for r in rows) for s in sensors}
    maxs = {s: max(r[s] for r in rows) for s in sensors}
    avgs = {s: sum(r[s] for r in rows) / len(rows) for s in sensors}

    # Calculate uptime from first to last reading
    try:
        first_ts = datetime.strptime(rows[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
        last_ts = datetime.strptime(rows[-1]["timestamp"], "%Y-%m-%d %H:%M:%S")
        uptime = (last_ts - first_ts).total_seconds() / 3600
    except (ValueError, IndexError):
        uptime = 0

    # Total readings = count all rows in CSV (not just filtered)
    total = 0
    try:
        with open(CSV_PATH, "r") as f:
            total = sum(1 for _ in f) - 1  # subtract header
    except Exception:
        total = len(rows)

    return {
        "total_readings": max(total, 0),
        "uptime_hours": round(uptime, 2),
        "min": mins,
        "max": maxs,
        "avg": avgs,
    }




def archive_old_data():
    """Move data older than DATA_RETENTION_DAYS to archive files."""
    data_path = Path(CSV_PATH)
    if not data_path.exists():
        return

    cutoff = datetime.now() - timedelta(days=DATA_RETENTION_DAYS)
    keep_rows = []
    archive_rows = []

    try:
        with open(data_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                try:
                    ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    if ts >= cutoff:
                        keep_rows.append(row)
                    else:
                        archive_rows.append(row)
                except (ValueError, KeyError):
                    keep_rows.append(row)
    except Exception as e:
        print(f"Archive error reading: {e}")
        return

    if not archive_rows:
        return

    # Write archive
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    archive_file = os.path.join(
        ARCHIVE_DIR,
        f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    try:
        with open(archive_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(archive_rows)

        # Rewrite main CSV with only recent data
        with open(data_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(keep_rows)

        print(f"Archived {len(archive_rows)} rows to {archive_file}")
    except Exception as e:
        print(f"Archive error writing: {e}")


# Run archival check on startup and every 6 hours
def schedule_archival():
    archive_old_data()
    timer = threading.Timer(6 * 3600, schedule_archival)
    timer.daemon = True
    timer.start()


# ── PULL FROM PI ──

def _get_last_csv_timestamp():
    """Read the timestamp of the last row in the local CSV."""
    path = Path(CSV_PATH)
    if not path.exists():
        return None
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            last = None
            for row in reader:
                last = row.get("timestamp")
            return last
    except Exception:
        return None


def _append_pi_reading(reading):
    """Append a reading dict (with timestamp) to the local CSV, skipping duplicates."""
    ts = reading.get("timestamp", "")
    if not ts:
        return False

    last_ts = _get_last_csv_timestamp()
    if last_ts and ts <= last_ts:
        return False  # duplicate or older

    # Ensure CSV exists
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "temperature", "turbidity", "ph", "conductivity"])

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts,
            reading.get("temperature", 0),
            reading.get("turbidity", 0),
            reading.get("ph", 0),
            reading.get("conductivity", 0),
        ])
    return True


def pull_from_pi():
    """Fetch the latest reading from the Pi's API and append to local CSV."""
    if not PI_SENSOR_URL:
        return
    url = f"{PI_SENSOR_URL.rstrip('/')}/api/latest"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if _append_pi_reading(data):
            print(
                f"[Pi pull] {data.get('timestamp', '?')}  "
                f"T={data.get('temperature', '?')}  "
                f"Turb={data.get('turbidity', '?')}  "
                f"pH={data.get('ph', '?')}  "
                f"Cond={data.get('conductivity', '?')}"
            )
        else:
            pass  # duplicate, no log spam
    except Exception as e:
        print(f"[Pi pull] Error: {e}")


def schedule_pi_pull():
    """Pull from Pi on a recurring timer (daemon thread)."""
    pull_from_pi()
    timer = threading.Timer(PI_PULL_INTERVAL, schedule_pi_pull)
    timer.daemon = True
    timer.start()


# ── ROUTES ──

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/data")
def api_data():
    time_range = request.args.get("range", "6h")
    if time_range not in RANGE_MAP:
        time_range = "6h"

    rows = read_csv_data(time_range)

    if not rows:
        return jsonify({
            "current": {"temperature": 0, "turbidity": 0, "ph": 0, "conductivity": 0},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "history": {"timestamps": [], "temperature": [], "turbidity": [], "ph": [], "conductivity": []},
            "stats": {"total_readings": 0, "uptime_hours": 0, "min": None, "max": None, "avg": None},
            "bloom_risk": {"bloom": False},
        })

    current = {
        "temperature": rows[-1]["temperature"],
        "turbidity": rows[-1]["turbidity"],
        "ph": rows[-1]["ph"],
        "conductivity": rows[-1]["conductivity"],
    }

    history = {
        "timestamps": [r["timestamp"] for r in rows],
        "temperature": [r["temperature"] for r in rows],
        "turbidity": [r["turbidity"] for r in rows],
        "ph": [r["ph"] for r in rows],
        "conductivity": [r["conductivity"] for r in rows],
    }

    stats = compute_stats(rows)

    return jsonify({
        "current": current,
        "timestamp": rows[-1]["timestamp"],
        "history": history,
        "stats": stats,
        "bloom_risk": {"bloom": False},
    })


@app.route("/api/reading", methods=["POST"])
def api_reading():
    """Receive a sensor reading via POST and append it to the CSV."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    try:
        reading = {
            "temperature": float(data["temperature"]),
            "turbidity": float(data["turbidity"]),
            "ph": float(data["ph"]),
            "conductivity": float(data["conductivity"]),
        }
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Missing or invalid field: {e}"}), 400

    # Ensure CSV exists with headers
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "temperature", "turbidity", "ph", "conductivity"])

    # Append the reading
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            reading["temperature"],
            reading["turbidity"],
            reading["ph"],
            reading["conductivity"],
        ])

    print(
        f"[{timestamp}] Received reading: "
        f"T={reading['temperature']} "
        f"Turb={reading['turbidity']} "
        f"pH={reading['ph']} "
        f"Cond={reading['conductivity']}"
    )

    return jsonify({"status": "ok", "timestamp": timestamp, **reading}), 201


if __name__ == "__main__":
    schedule_archival()
    if PI_SENSOR_URL:
        schedule_pi_pull()
    print("=" * 50)
    print("  Cyanobacteria Detection Dashboard")
    print(f"  Reading data from: {CSV_PATH}")
    print(f"  Pi sensor pull: {PI_SENSOR_URL or 'disabled'} (every {PI_PULL_INTERVAL}s)")
    print(f"  Data retention: {DATA_RETENTION_DAYS} days")
    print("  Dashboard: http://0.0.0.0:8000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8000, debug=False)
