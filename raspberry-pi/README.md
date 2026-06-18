# Raspberry Pi Deployment

This directory contains the two scripts that run on (or alongside) the Raspberry Pi.

| Script | Role |
|---|---|
| `sensor_logger.py` | Reads the four water-quality sensors every 10 s and appends readings to `data/sensor_data.csv`. Also exposes a lightweight HTTP API so the dashboard can pull the latest reading. |
| `dashboard.py` | Flask web server that reads `data/sensor_data.csv` and serves a real-time monitoring dashboard with live charts, summary stats, and bloom-risk status. |

---

## Wiring

| Sensor | Model | Pi Connection |
|---|---|---|
| Temperature | DS18B20 | 1-Wire — GPIO4 |
| Turbidity | DFRobot SEN0189 | ADS1115 channel A0 (I2C) |
| pH | DFRobot SEN0161-V2 | ADS1115 channel A2 (I2C) |
| Conductivity | DFRobot SEN0244 | ADS1115 channel A3 (I2C) |
| ADC | DFR0553 / ADS1115 | I2C address 0x48 |

Enable 1-Wire and I2C in `raspi-config` before running.

---

## Setup

```bash
pip install flask w1thermsensor adafruit-circuitpython-ads1x15
```

---

## Running

**Sensor logger** (on the Pi):

```bash
# Live hardware
python raspberry-pi/sensor_logger.py

# Simulated data (no hardware required)
python raspberry-pi/sensor_logger.py --simulate

# Custom options
python raspberry-pi/sensor_logger.py --interval 30 --api-port 8000 --dashboard-url http://192.168.1.100:8000
```

**Dashboard** (Pi or any machine on the same network):

```bash
python raspberry-pi/dashboard.py
```

Then visit `http://<host-ip>:8000`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SENSOR_CSV_PATH` | `../data/sensor_data.csv` | Override the CSV path |
| `PI_SENSOR_URL` | `http://192.168.68.111:8000` | Pi's API address (used by dashboard to pull readings) |
| `DASHBOARD_URL` | *(none)* | Dashboard URL for the logger to POST readings to |

---

## Calibration Notes

- **pH:** Calibrate with pH 7 buffer solution and adjust `offset` in `read_ph()`.
- **Turbidity:** DFRobot SEN0189 calibration curve is pre-loaded; verify with a known NTU standard.
- **Conductivity:** Temperature compensation is applied automatically using the live temperature reading.
