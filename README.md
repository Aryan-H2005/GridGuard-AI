# ⚡ GridGuard AI

GridGuard AI is an unsupervised machine learning system that detects unusual electricity consumption patterns.

## Problem

Electricity providers may need to identify unusual consumption patterns that could require further investigation.

## Solution

GridGuard AI uses the Isolation Forest algorithm to identify anomalous electricity consumption readings.

## Features

- Electricity consumption analysis
- Historical consumption features
- Anomaly detection
- Risk scoring
- Risk classification
- Consumer-wise analysis
- Suspicious consumer identification
- Interactive Streamlit dashboard
- CSV report download

## Machine Learning

The project uses Isolation Forest because labeled electricity theft data is difficult to obtain.

### Features Used

- consumption_kwh
- previous_consumption
- consumption_change
- percentage_change
- avg_consumption
- std_consumption
- min_consumption
- max_consumption

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Isolation Forest
- Plotly
- Streamlit
- Joblib

## Dashboard

The dashboard provides:

- Total consumers
- Total readings
- Suspicious readings
- High-risk readings
- Suspicious consumers
- Consumption distribution
- Monthly consumption trends
- Risk distribution
- Consumer-level analysis

## Important Note

A suspicious reading does not confirm electricity theft.

The system identifies unusual consumption patterns that may require further investigation.

## How to Run

```bash
python src/data_generator.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py
streamlit run app.py
```
