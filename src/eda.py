import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/electricity_data.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\n Dataset shape:", df.shape)

print("\nColumn name:", df.columns)

print("\n Dataset Information:", df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate values
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Unique consumers
print("\nNumber of Consumers:")
print(df["consumer_id"].nunique())

# Average consumption
print("\nAverage Consumption:")
print(df["consumption_kwh"].mean())

# Minimum consumption
print("\nMinimum Consumption:")
print(df["consumption_kwh"].min())

# Maximum consumption
print("\nMaximum Consumption:")
print(df["consumption_kwh"].max())

#Distribution of consumption
plt.figure(figsize=(8, 5))
sns.histplot(df["consumption_kwh"], bins=40, kde=True)
plt.title("Distribution of Electricity Consumption")
plt.xlabel("Consumption (kWh)")
plt.ylabel("Frequency")
plt.show()

#Monthly consumption trends
monthly_consumption = df.groupby("month")["consumption_kwh"].mean()
plt.figure(figsize=(8, 5))
monthly_consumption.plot(marker='o')
plt.title("Average Monthly Electricity Consumption")
plt.xlabel("Month")
plt.ylabel("Average Consumption (kWh)")
plt.grid()
plt.show()

# 3. Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["consumption_kwh"])
plt.title("Electricity Consumption Outliers")
plt.xlabel("Consumption (kWh)")
plt.show()


# 4. Sample consumer consumption
consumer = 1

consumer_data = df[df["consumer_id"] == consumer]

plt.figure(figsize=(8, 5))
plt.plot(
    consumer_data["month"],
    consumer_data["consumption_kwh"],
    marker="o"
)

plt.title(f"Consumer {consumer} Monthly Consumption")
plt.xlabel("Month")
plt.ylabel("Consumption (kWh)")
plt.grid()
plt.show()
