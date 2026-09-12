import pandas as pd

df = pd.read_csv("data/predictions.csv")

print("===== GridGuard AI Model Evaluation =====")

print("\nTotal Readings:")
print(len(df))

print("\nNormal Readings:")
print(len(df[df["status"] == "Normal"]))

print("\nSuspicious Readings:")
print(len(df[df["status"] == "Suspicious"]))

print("\nSuspicious Percentage:")

suspicious_percentage = (
    len(df[df["status"] == "Suspicious"]) /
    len(df)
) * 100

print(round(suspicious_percentage, 2), "%")


print("\n===== Risk Distribution =====")

print(
    df["risk_level"].value_counts()
)


print("\n===== Average Risk Score =====")

print(
    round(df["risk_score"].mean(), 2)
)


print("\n===== Top 10 Suspicious Readings =====")

top_suspicious = df.sort_values(
    "risk_score",
    ascending=False
).head(10)

print(
    top_suspicious[
        [
            "consumer_id",
            "month",
            "consumption_kwh",
            "percentage_change",
            "risk_score",
            "risk_level",
            "status"
        ]
    ]
)