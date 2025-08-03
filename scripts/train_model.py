import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

df = pd.read_csv("data/features.csv")
df.dropna(inplace=True)

X = df[["hour", "day", "month", "weekday", "aqius", "aqius_change"]]
y = df["aqius_future"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/aqi_model.pkl")
print("Model trained and saved.")
