import pandas as pd

df = pd.read_csv("titanic.csv")

print("--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info & Data Types ---")
print(df.info())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Survival Rate by Gender ---")
print(df.groupby("Sex")["Survived"].mean())
import matplotlib.pyplot as plt

# 1. Data Cleaning
df["Age"] = df["Age"].fillna(df["Age"].median())  # Fill missing ages with median
df = df.drop(columns=["Cabin"])                   # Drop Cabin (too many missing values)

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

# 2. Plot Survival Count by Gender
df.groupby(["Sex", "Survived"]).size().unstack().plot(kind="bar", color=["salmon", "skyblue"])
plt.title("Titanic Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.legend(["Did Not Survive (0)", "Survived (1)"])
plt.tight_layout()
plt.show()
# 3. Plot Survival Count by Passenger Class (Pclass)
df.groupby(["Pclass", "Survived"]).size().unstack().plot(kind="bar", color=["salmon", "skyblue"])
plt.title("Titanic Survival Count by Passenger Class")
plt.xlabel("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.legend(["Did Not Survive (0)", "Survived (1)"])
plt.tight_layout()
plt.show()

# 4. Plot Age Distribution
df["Age"].plot(kind="hist", bins=8, color="teal", edgecolor="black")
plt.title("Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()