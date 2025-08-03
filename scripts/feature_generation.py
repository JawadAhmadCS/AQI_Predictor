import pandas as pd
from datetime import datetime

df = pd.read_csv("data/weather_init.csv")

df.columns = [
    "timestamp", "city", "state", "country", "lat", "lon",
    "aqius", "mainus", "aqicn", "maincn", "temperature", "humidity",
    "pressure", "wind_direction", "wind_speed"
]

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday

df['aqius_change'] = df['aqius'].diff().fillna(0)
df['aqius_future'] = df['aqius'].shift(-3)  #after 3 hours

df.to_csv("data/features.csv", index=False)
print("Features generated and saved.")
