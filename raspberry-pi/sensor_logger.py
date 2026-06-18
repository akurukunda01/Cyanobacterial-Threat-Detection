"""
Cyanobacteria Detection System - Sensor Logger
Reads sensors every 10 seconds and appends data to CSV.
Optionally exposes a built-in HTTP API for remote access.

Hardware:
  - DS18B20 Temperature sensor (1-Wire, GPIO4)
  - DFRobot SEN0189 Turbidity via DFR0553 ADC (I2C, A0)
  - DFRobot SEN0161-V2 pH via DFR0553 ADC (I2C, A2)
  - DFRobot SEN0244 TDS/Conductivity via DFR0553 ADC (I2C, A3)
  - DFR0553 ADC module (ADS1115, I2C address 0x48)

Usage:
  python3 sensor_logger.py              # Normal mode (requires sensors)
  python3 sensor_logger.py --simulate   # Simulated data for testing
  python3 sensor_logger.py --api-port 0 # Disable built-in API server
"""

import os
import csv
import sys
import time
import json
import random
import argparse
import threading
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request as flask_request

_PI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PI_DIR)
CSV_PATH = os.environ.get(
    "SENSOR_CSV_PATH",
    os.path.join(_PROJECT_ROOT, "data", "sensor_data.csv")
)
INTERVAL = 10  # seconds between readings


# ── REAL SENSOR READING ──

def init_sensors():
    """Initialize sensor hardware. Returns sensor objects."""
    try:
        from w1thermsensor import W1ThermSensor
        import board
        import busio
        import adafruit_ads1x15.ads1115 as ADS
        from adafruit_ads1x15.analog_in import AnalogIn

        temp_sensor = W1ThermSensor()
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c)

        turbidity_ch = AnalogIn(ads, ADS.P0)    # A0
        ph_ch = AnalogIn(ads, ADS.P2)            # A2
        conductivity_ch = AnalogIn(ads, ADS.P3)  # A3

        print("Sensors initialized successfully.")
        return {
            "temp": temp_sensor,
            "turbidity": turbidity_ch,
            "ph": ph_ch,
            "conductivity": conductivity_ch,
        }
    except Exception as e:
        print(f"Sensor init failed: {e}")
        print("Run with --simulate flag for testing without hardware.")
        sys.exit(1)


def read_temperature(sensor):
    """Read DS18B20 temperature in Celsius."""
    return sensor.get_temperature()


def read_turbidity(channel):
    """Convert turbidity sensor voltage to NTU."""
    voltage = channel.voltage
    # DFRobot SEN0189 calibration curve (voltage to NTU)
    # At 4.2V = 0 NTU (clear), decreasing voltage = higher turbidity
    if voltage >= 4.2:
        return 0.0
    elif voltage <= 2.5:
        return 3000.0
    else:
        ntu = -1120.4 * (voltage ** 2) + 5742.3 * voltage - 4352.9
        return max(0.0, ntu)


def read_ph(channel):
    """Convert pH sensor voltage to pH value."""
    voltage = channel.voltage
    # DFRobot SEN0161-V2 calibration
    # Neutral (pH 7) ~= 2.5V, slope ~= -5.7 pH/V
    # Adjust offset and slope based on your calibration solution results
    offset = 0.0  # Calibration offset (adjust after calibrating with pH 7 buffer)
    slope = -5.7  # mV per pH unit
    ph_value = 7.0 + ((2.5 - voltage) / (3.3 / 14.0)) + offset
    return max(0.0, min(14.0, ph_value))


def read_conductivity(channel, temperature):
    """Convert TDS sensor voltage to conductivity (µS/cm) with temp compensation."""
    voltage = channel.voltage
    # DFRobot SEN0244 formula
    # Temperature compensation coefficient
    compensation = 1.0 + 0.02 * (temperature - 25.0)
    compensated_voltage = voltage / compensation
    # TDS in ppm
    tds = (133.42 * compensated_voltage ** 3
           - 255.86 * compensated_voltage ** 2
           + 857.39 * compensated_voltage) * 0.5
    # Convert TDS (ppm) to conductivity (µS/cm): approx factor of 2
    conductivity = tds * 2.0
    return max(0.0, conductivity)


def read_all_sensors(sensors):
    """Read all sensors and return dict of values."""
    temp = read_temperature(sensors["temp"])
    return {
        "temperature": round(temp, 2),
        "turbidity": round(read_turbidity(sensors["turbidity"]), 2),
        "ph": round(read_ph(sensors["ph"]), 2),
        "conductivity": round(read_conductivity(sensors["conductivity"], temp), 2),
    }


# ── SIMULATED DATA ──

class SimulatedSensors:
    """Generates realistic-looking sensor data for testing."""

    def __init__(self):
        self.temp = 18.0 + random.uniform(-2, 2)
        self.turb = 40.0 + random.uniform(-5, 5)
        self.ph = 7.2 + random.uniform(-0.3, 0.3)
        self.cond = 500.0 + random.uniform(-50, 50)

    def read(self):
        # Random walk with drift and bounds
        self.temp += random.gauss(0, 0.15)
        self.temp = max(5, min(35, self.temp))

        self.turb += random.gauss(0, 1.0)
        self.turb = max(0, min(500, self.turb))

        self.ph += random.gauss(0, 0.02)
        self.ph = max(4, min(10, self.ph))

        self.cond += random.gauss(0, 5.0)
        self.cond = max(50, min(2000, self.cond))

        return {
            "temperature": round(self.temp, 2),
            "turbidity": round(self.turb, 2),
            "ph": round(self.ph, 2),
            "conductivity": round(self.cond, 2),
        }


# ── CSV LOGGING ──

def ensure_csv(path):
    """Create CSV file with headers if it doesn't exist."""
    if not Path(path).exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "temperature", "turbidity", "ph", "conductivity"])
        print(f"Created {path}")


def append_reading(path, reading):
    """Append a sensor reading to the CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            reading["temperature"],
            reading["turbidity"],
            reading["ph"],
            reading["conductivity"],
        ])
    return timestamp


# ── BUILT-IN FLASK API SERVER ──

def _read_csv_tail(csv_path, n=1):
    """Read the last N rows from the CSV file, returned as list of dicts."""
    path = Path(csv_path)
    if not path.exists():
        return []
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[-n:] if n < len(rows) else rows
    except Exception:
        return []


def _create_api_app(csv_path):
    """Create a Flask app that serves the sensor CSV data."""
    api_app = Flask(__name__)

    @api_app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @api_app.route("/api/latest")
    def api_latest():
        n = flask_request.args.get("n", 1, type=int)
        n = max(1, n)
        rows = _read_csv_tail(csv_path, n)
        if not rows:
            return jsonify({"error": "no data"}), 404
        # Convert numeric strings to floats
        result = []
        for row in rows:
            result.append({
                "timestamp": row.get("timestamp", ""),
                "temperature": float(row.get("temperature", 0)),
                "turbidity": float(row.get("turbidity", 0)),
                "ph": float(row.get("ph", 0)),
                "conductivity": float(row.get("conductivity", 0)),
            })
        if n == 1:
            return jsonify(result[0])
        return jsonify(result)

    return api_app


def start_api_server(csv_path, port):
    """Start the Flask API server in a background daemon thread."""
    api_app = _create_api_app(csv_path)
    thread = threading.Thread(
        target=lambda: api_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return api_app


# ── DASHBOARD POST ──

def send_to_dashboard(url, reading):
    """POST a reading to the dashboard API. Logs result, never raises."""
    endpoint = f"{url.rstrip('/')}/api/reading"
    data = json.dumps(reading).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"  -> POST OK ({resp.status})")
    except urllib.error.HTTPError as e:
        print(f"  -> POST failed: HTTP {e.code} {e.reason}")
    except Exception as e:
        print(f"  -> POST failed: {e}")


# ── MAIN LOOP ──

def main():
    parser = argparse.ArgumentParser(description="Cyanobacteria Sensor Logger")
    parser.add_argument(
        "--simulate", action="store_true",
        help="Use simulated sensor data (for testing without hardware)"
    )
    parser.add_argument(
        "--csv", type=str, default=CSV_PATH,
        help=f"CSV output path (default: {CSV_PATH})"
    )
    parser.add_argument(
        "--interval", type=int, default=INTERVAL,
        help=f"Seconds between readings (default: {INTERVAL})"
    )
    parser.add_argument(
        "--dashboard-url", type=str,
        default=os.environ.get("DASHBOARD_URL"),
        help="Dashboard URL to POST readings to (e.g. http://192.168.1.100:8000)"
    )
    parser.add_argument(
        "--api-port", type=int, default=8000,
        help="Port for built-in HTTP API server (0 to disable, default: 8000)"
    )
    args = parser.parse_args()

    csv_path = args.csv
    dashboard_url = args.dashboard_url
    ensure_csv(csv_path)

    # Start built-in API server
    if args.api_port > 0:
        try:
            start_api_server(csv_path, args.api_port)
            api_info = f"http://0.0.0.0:{args.api_port}"
        except OSError as e:
            print(f"Warning: Could not start API server on port {args.api_port}: {e}")
            api_info = "failed to start"
    else:
        api_info = "disabled"

    print("=" * 50)
    print("  Cyanobacteria Sensor Logger")
    print(f"  Mode: {'SIMULATED' if args.simulate else 'LIVE SENSORS'}")
    print(f"  CSV: {csv_path}")
    print(f"  Interval: {args.interval}s")
    print(f"  API: {api_info}")
    print(f"  Dashboard: {dashboard_url or 'disabled'}")
    print("=" * 50)

    if args.simulate:
        sim = SimulatedSensors()
        read_fn = sim.read
    else:
        sensors = init_sensors()
        read_fn = lambda: read_all_sensors(sensors)

    reading_count = 0
    try:
        while True:
            reading = read_fn()
            ts = append_reading(csv_path, reading)
            reading_count += 1
            if dashboard_url:
                send_to_dashboard(dashboard_url, reading)
            print(
                f"[{ts}] #{reading_count:>5}  "
                f"T={reading['temperature']:6.2f}°C  "
                f"Turb={reading['turbidity']:7.2f}NTU  "
                f"pH={reading['ph']:5.2f}  "
                f"Cond={reading['conductivity']:7.2f}µS/cm"
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nStopped. {reading_count} readings logged to {csv_path}")


if __name__ == "__main__":
    main()
