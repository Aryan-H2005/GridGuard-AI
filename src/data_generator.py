import pandas as pd
import numpy as np

np.random.seed(42)

num_consumers = 1000
months = 12

data = []

for consumer_id in range(1, num_consumers + 1):

    base_usage = np.random.randint(100, 500)

    for month in range(1, months + 1):

        consumption = base_usage + np.random.normal(0, 30)

        data.append([
            consumer_id,
            month,
            round(max(consumption, 20), 2)
        ])

df = pd.DataFrame(
    data,
    columns=[
        "consumer_id",
        "month",
        "consumption_kwh"
    ]
)

# Add abnormal consumption patterns
anomaly_indices = np.random.choice(
    len(df),
    size=int(len(df) * 0.05),
    replace=False
)

df.loc[anomaly_indices, "consumption_kwh"] *= np.random.uniform(
    0.2, 0.5, len(anomaly_indices)
)

df["consumption_kwh"] = df["consumption_kwh"].round(2)

df.to_csv(
    "data/electricity_data.csv",
    index=False
)

print("Dataset created successfully!")
print("Shape:", df.shape)
print(df.head())