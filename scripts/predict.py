import pandas as pd
import joblib
import calendar

df = pd.read_csv("data/features.csv").dropna()
last_row = df.iloc[-1:]

model = joblib.load("models/aqi_model.pkl")

X = last_row[["hour", "day", "month", "weekday", "aqius", "aqius_change"]].copy()

predictions = []

# 72 hours = 3 days
for i in range(72):
    pred = model.predict(X)[0]
    predictions.append(round(pred, 2))

    # Update values for next hour
    current_hour = int(X["hour"].values[0])
    current_day = int(X["day"].values[0])
    current_month = int(X["month"].values[0])
    current_weekday = int(X["weekday"].values[0])
    current_aqius = float(X["aqius"].values[0])

    # Update hour
    next_hour = (current_hour + 1) % 24
    X["hour"] = next_hour

    # Agar 23 -> 0 ho to day bhi +1
    if next_hour == 0:
        # Din increase
        days_in_month = calendar.monthrange(2025, current_month)[1]  # 2025 replace with your data year
        next_day = current_day + 1
        next_weekday = (current_weekday + 1) % 7

        # Agar mahine ka last din cross ho jaye
        if next_day > days_in_month:
            next_day = 1
            next_month = current_month + 1
            if next_month > 12:
                next_month = 1
        else:
            next_month = current_month
        X["day"] = next_day
        X["month"] = next_month
        X["weekday"] = next_weekday

    # Update AQI values
    X["aqius_change"] = pred - current_aqius
    X["aqius"] = pred

print("AQI Predictions for next 3 days:", predictions)
