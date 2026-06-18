# Cyanobacterial Threat Detection

**Aadi Kurukunda**

An end-to-end system for early detection of harmful cyanobacterial (blue-green algae) blooms in freshwater lakes using machine learning and a Raspberry Pi sensor array.
<img width="4608" height="3456" alt="Cyanobacteria Detection Board (1)" src="https://github.com/user-attachments/assets/17960b3a-c0bc-438d-a97b-10862a66c5b6" />


---

## Overview

Comprehensive cyanobacterial bloom monitoring currently requires specialized lab equipment and trained personnel — making it prohibitively expensive for many at-risk communities. This project demonstrates that machine learning trained on just four low-cost water quality sensors (temperature, pH, turbidity, conductivity) can provide effective early-warning bloom detection at a fraction of the cost.

ML models were trained on ~1,000 U.S. lakes from the EPA National Lakes Assessment 2017, then deployed on a Raspberry Pi mounted on a stationary floating platform and field-tested on a real water body — all for under $300.

**Key results:** The best model (Stacking Ensemble: Logistic Regression + XGBoost + Neural Network) achieved a **ROC-AUC of 0.7723** and **87.9% recall**, successfully detecting the majority of bloom conditions using only the four low-cost sensors. This confirmed the project hypothesis (ROC-AUC > 0.75), demonstrating feasibility for accessible, community-deployable monitoring. Full methodology and results are documented in `cyano_paper.pdf`.

---

## Repository Structure

```
.
├── CynobacteriaThreatDetection.ipynb   # Main ML notebook (model training & analysis)
├── cyano_paper.pdf                     # Research paper
├── cyano_photo1.jpg                    # Reference bloom photograph
│
├── data/                               # Datasets
│   ├── nla_2017_phytoplankton_count-data.csv   # EPA NLA 2017 phytoplankton biovolume
│   ├── nla_2017_profile-data.csv               # Water column sensor profiles
│   ├── nla_2017_water_chemistry_chla-data.csv  # Turbidity & water chemistry
│   ├── cyano_prediction_data.csv               # Processed regression dataset
│   ├── cyano_prediction_data_with_turb.csv     # Processed dataset (with turbidity)
│   ├── cyano_classification_data_threshold_20.csv  # Binary bloom classification dataset
│   └── sensor_data.csv                         # Live sensor data (written by Pi)
│
├── src/
│   └── data.py                         # Data processing & ML training scripts
│
├── raspberry-pi/                       # Raspberry Pi deployment
│   ├── sensor_logger.py                # Reads hardware sensors → logs to CSV
│   └── dashboard.py                    # Flask dashboard that visualizes live data
│
└── assets/
    └── Cyanobacteria Detection Board (1).png   # Project overview diagram
```

---

## How It Works

```
[ Floating Platform ]
  PVC-pipe pontoons + waterproof junction box
  Raspberry Pi + four low-cost water quality sensors
         |
         v
[ sensor_logger.py ]  reads sensors every 10 s  ──→  data/sensor_data.csv
         |
         v  (POST every 10 s)
[ dashboard.py ]  ──→  Web dashboard (http://<pi-ip>:8000)
                         Live charts, stats, bloom risk indicator
```

The ML model — trained in `CynobacteriaThreatDetection.ipynb` on 941 U.S. lakes from the EPA NLA 2017 survey — runs on the Pi and classifies whether the current sensor readings indicate a bloom-risk condition. A decision threshold of 0.35 (rather than the default 0.5) is used to prioritize recall: in water quality monitoring, missing an actual bloom is more dangerous than a false alarm.

---

## Physical Platform

The system was built as a stationary floating monitoring station for real-world field deployment:

- **Float:** PVC pipe pontoons with a waterproof junction box housing the Raspberry Pi and sensor electronics
- **Sensors:** Four low-cost probes for temperature, pH, turbidity, and conductivity (see Hardware section)
- **Connectivity:** Pi runs `sensor_logger.py` locally and streams readings to `dashboard.py` in real time
- **Total build cost: ~$300** — compared to thousands of dollars for conventional lab-based bloom monitoring
- **Field tested** on a local water body after validation in a controlled indoor environment

---

## ML Approach

**Features:** Surface temperature (°C), pH, turbidity (NTU), conductivity (µS/cm)  
**Target:** Binary bloom classification (>10% cyanobacterial biovolume = bloom, per EPA early-detection guidelines)  
**Training data:** 941 lakes from EPA NLA 2017 — phytoplankton biovolume counts merged with water-column sensor profiles  
**Train/test split:** 80% / 20% stratified (752 / 189 lakes)

Five algorithms were trained and progressively optimized through three stages:

| Stage | Avg. ROC-AUC gain |
|---|---|
| Baseline (default params) | — |
| + Hyperparameter tuning (GridSearchCV / RandomizedSearchCV, 5-fold CV) | +~8% |
| + Feature engineering (6 engineered features from sensor biology) | +~2% |
| + Ensemble methods (stacking & soft voting combinations) | marginal |

**Final model results (after all optimization, decision threshold = 0.35):**

| Model | ROC-AUC | Recall | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.7604 | 0.868 | 0.699 |
| Random Forest | 0.7487 | 0.879 | 0.714 |
| Gradient Boosting | 0.7608 | 0.868 | 0.715 |
| Neural Network (MLP) | 0.7587 | 0.769 | 0.704 |
| XGBoost | 0.7649 | 0.868 | 0.699 |
| **Stacking Ensemble (LR + XGB + MLP)** | **0.7723** | **0.879** | — |

**Top predictors (Random Forest feature importance):** Turbidity (0.172), pH (0.155), Temp×pH interaction (0.154)

For full methodology, results tables, figures, and discussion see `cyano_paper.pdf`. For code walkthroughs and visualizations (ROC curves, confusion matrix, feature distributions) see `CynobacteriaThreatDetection.ipynb`.

---

## Quick Start

### 1. Reproduce the ML Results

Open the notebook in Jupyter:

```bash
jupyter notebook CynobacteriaThreatDetection.ipynb
```

To re-run the data processing pipeline from the raw NLA files:

```bash
python src/data.py
```

### 2. Raspberry Pi Deployment

**Requirements:** Raspberry Pi 3/4/5, sensors listed below, Python 3.9+

Install dependencies on the Pi:

```bash
pip install flask w1thermsensor adafruit-circuitpython-ads1x15
```

**Step 1 — Start the sensor logger** (reads hardware every 10 s):

```bash
python raspberry-pi/sensor_logger.py
```

Use `--simulate` for testing without physical sensors:

```bash
python raspberry-pi/sensor_logger.py --simulate
```

**Step 2 — Start the dashboard** (on the same or a separate machine):

```bash
python raspberry-pi/dashboard.py
```

Open `http://<pi-ip-address>:8000` in a browser to view the live dashboard.

#### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SENSOR_CSV_PATH` | `data/sensor_data.csv` | Path to the sensor data CSV |
| `PI_SENSOR_URL` | `http://192.168.68.111:8000` | URL of Pi's sensor API (set on dashboard host) |
| `DASHBOARD_URL` | *(disabled)* | Dashboard URL for the Pi to POST readings to |

---

## Hardware

| Sensor | Model | Interface |
|---|---|---|
| Temperature | DS18B20 | 1-Wire (GPIO4) |
| Turbidity | DFRobot SEN0189 | Analog via ADS1115 (A0) |
| pH | DFRobot SEN0161-V2 | Analog via ADS1115 (A2) |
| Conductivity / TDS | DFRobot SEN0244 | Analog via ADS1115 (A3) |
| ADC | DFR0553 / ADS1115 | I2C (0x48) |

---

## Data Sources

- **EPA National Lakes Assessment 2017** — phytoplankton counts, water-column profiles, and water chemistry from 1,000+ U.S. lakes.  
  Published by the U.S. Environmental Protection Agency.

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
flask
jupyter
```

Pi-only (hardware): `w1thermsensor`, `adafruit-circuitpython-ads1x15`, `board`, `busio`
