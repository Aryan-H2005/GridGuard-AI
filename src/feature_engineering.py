import pandas as pd

# Load data
df = pd.read_csv("data/electricity_data.csv")

# Sort data
df = df.sort_values(["consumer_id", "month"])

# Create consumer-level features
features = df.groupby("consumer_id")["consumption_kwh"].agg(
    avg_consumption="mean",
    std_consumption="std",
    min_consumption="min",
    max_consumption="max"
).reset_index()

# Calculate consumption change
df["previous_consumption"] = df.groupby(
    "consumer_id"
)["consumption_kwh"].shift(1)

df["consumption_change"] = (
    df["consumption_kwh"] - df["previous_consumption"]
)

# Calculate percentage change
df["percentage_change"] = (
    df["consumption_change"] /
    df["previous_consumption"]
) * 100

# Remove first month because it has no previous value
df = df.dropna()

# Merge consumer-level features
df = df.merge(features, on="consumer_id")

# Save feature dataset
df.to_csv(
    "data/feature_data.csv",
    index=False
)

print("Feature dataset created!")
print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())