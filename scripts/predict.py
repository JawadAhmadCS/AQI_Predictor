import pandas as pd
import joblib

df = pd.read_csv("data/features.csv").dropna()
last_row = df.iloc[-1:]

model = joblib.load("models/aqi_model.pkl")

X = last_row[["hour", "day", "month", "weekday", "aqius", "aqius_change"]]

predictions = []
for i in range(3):
    pred = model.predict(X)[0]
    predictions.append(round(pred, 2))
    X["aqius"] = pred
    X["aqius_change"] = pred - X["aqius"].values[0]
    X["hour"] = (X["hour"] + 1) % 24

print("AQI Predictions for next 3 hours:", predictions)
