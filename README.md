# Time-Series Framework for Urban Flood Risk Assessment (FPI)

This repository provides the Python codes used in:

Kim, S.Y. (2026).  
Time-Series Framework Integrating Rainfall and Satellite Soil Moisture  
for Multi-Scale Urban Flood Risk Assessment.  
International Journal of Applied Earth Observation and Geoinformation.

## Overview

This framework integrates:

- Hourly rainfall (Korea Meteorological Administration, KMA)
- 3-hourly NASA SMAP Level 4 soil moisture (SPL4SMGP)
- Dynamic imperviousness derived from surface–root zone time lag

to compute a Flood Potential Index (FPI).

The framework evaluates 27 multi-time window combinations and
quantifies:

- Duration above top 95% threshold
- Peak Intensity Ratio (PIR)
- Lead Time


---
## Repository Structure
├── analysis.py # Duration, PIR, Lead time analysis
├── FPI.py # FPI computation module
├── timeseries.py # Rainfall & soil moisture preprocessing
├── timeseries_FPI.py # FPI time-series generation
├── requirements.txt
├── LICENSE
└── README.md

## Data Sources

### 1. Rainfall Data
- Korea Meteorological Administration (KMA)
- Access via: https://data.kma.go.kr
- Temporal resolution: Hourly

### 2. Soil Moisture Data
- NASA SMAP Level 4 (SPL4SMGP, Version 008)
- Access via NASA Earthdata:
  https://earthdata.nasa.gov/

Users must download raw data independently according to their access permissions.

---

## How to Reproduce the Analysis

### Step 1. Preprocess time-series
python timeseries.py
### Step 2. Compute FPI time series
python timeseries_FPI.py
### Step 3. Evaluate performance metrics
python analysis.py

Output figures:
- Duration box plots & heatmaps
- PIR box plots & heatmaps
- Lead time box plots & heatmaps

---

## Python Environment

Recommended:
- Python 3.10+

Install dependencies: pip install -r requirements.txt

---

## License

This project is released under the MIT License.
