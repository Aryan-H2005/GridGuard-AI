import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest

# Load feature dataset
df = pd.read_csv("data/feature_data.csv")

# Fill missing values
df = df.fillna(0)

# Features used by the model
features = [
    "consumption_kwh",
    "consumption_change",
    "percentage_change",
    "avg_consumption",
    "std_consumption",
    "min_consumption",
    "max_consumption"
]

X = df[features]

# Create model
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

# Train model
model.fit(X)

# Predict anomalies
df["anomaly"] = model.predict(X)

# Anomaly score
df["anomaly_score"] = model.decision_function(X)

# Convert prediction
df["status"] = df["anomaly"].map({
    1: "Normal",
    -1: "Suspicious"
})

# Convert anomaly score to 0-100 risk score
min_score = df["anomaly_score"].min()
max_score = df["anomaly_score"].max()

df["risk_score"] = (
    (max_score - df["anomaly_score"]) /
    (max_score - min_score)
) * 100

df["risk_score"] = df["risk_score"].round(2)

# Risk level
def get_risk_level(score):

    if score < 30:
        return "Low"

    elif score < 70:
        return "Medium"

    else:
        return "High"


df["risk_level"] = df["risk_score"].apply(get_risk_level)

# Save model
joblib.dump(
    model,
    "models/isolation_forest.pkl"
)

# Save predictions
df.to_csv(
    "data/predictions.csv",
    index=False
)

print("Model trained successfully!")

print("\nStatus:")
print(df["status"].value_counts())

print("\nRisk Levels:")
print(df["risk_level"].value_counts())

print("\nTop suspicious readings:")

suspicious = df[
    df["status"] == "Suspicious"
].sort_values("risk_score", ascending=False)

print(
    suspicious[
        [
            "consumer_id",
            "month",
            "consumption_kwh",
            "percentage_change",
            "risk_score",
            "risk_level"
        ]
    ].head(10)
)