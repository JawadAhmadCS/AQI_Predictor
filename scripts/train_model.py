import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import numpy as np

# Load data
df = pd.read_csv("data/features.csv")
df.dropna(inplace=True)

# Features and target
X = df[["hour", "day", "month", "weekday", "aqius", "aqius_change"]]
y = df["aqius_future"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/aqi_model.pkl")
print("Model trained and saved to models/aqi_model.pkl")
