import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================
# 1. Load Dataset000
# ===============================
df = pd.read_csv("resource_usage_sample.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

print("\nDataset Loaded Successfully!\n")
print(df.head())


# ===============================
# 2. Feature Engineering
# ===============================
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["weekday"] = df["timestamp"].dt.weekday

# Target = CPU Demand
target = "cpu_utilization"

# Input Features
features = ["hour", "day", "weekday",
            "memory_utilization",
            "storage_utilization",
            "network_bandwidth"]

X = df[features]
y = df[target]


# ===============================
# 3. Train-Test Split (Time Series)
# ===============================
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]


# ===============================
# 4. Train Forecasting Model
# ===============================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# ===============================
# 5. Predict CPU Demand
# ===============================
y_pred = model.predict(X_test)


# ===============================
# 6. Evaluation Metrics
# ===============================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n Model Performance:")
print("MAE  =", mae)
print("RMSE =", rmse)


# ===============================
# 7. Visualization
# ===============================
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:50], label="Actual CPU Usage")
plt.plot(y_pred[:50], label="Predicted CPU Usage")
plt.title("CPU Demand Forecasting")
plt.xlabel("Time Step")
plt.ylabel("CPU Utilization (%)")
plt.legend()
plt.show()


# ===============================
# 8. Resource Allocation Optimization
# ===============================
print("\n Dynamic Resource Allocation Recommendations:\n")

for i in range(10):
    predicted_cpu = y_pred[i]

    if predicted_cpu > 80:
        decision = "Scale UP (Add more servers)"
    elif predicted_cpu < 30:
        decision = "Scale DOWN (Reduce servers)"
    else:
        decision = "Normal Allocation"

    print(f"Prediction {i+1}: CPU={predicted_cpu:.2f}% → {decision}")


# ===============================
# 9. Future Forecasting (Next 5 Hours)
# ===============================
print("\n Future CPU Demand Forecast (Next 5 Hours):\n")

future_data = X_test.iloc[-5:]
future_pred = model.predict(future_data)

for i, val in enumerate(future_pred):
    print(f"Hour +{i+1}: Predicted CPU Demand = {val:.2f}%")

print("\nDone Successfully!")