# scripts/getWeather.py

import requests
import csv
import os
from datetime import datetime

API_KEY = "d0880b1c-e430-4eba-8bcb-68c6605e7a6e"
city = "Islamabad"
state = "Islamabad"
country = "Pakistan"

url = f"https://api.airvisual.com/v2/city?city={city}&state={state}&country={country}&key={API_KEY}"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    if data["status"] != "success":
        print("API returned failure status.")
        exit()

    pollution = data["data"]["current"]["pollution"]
    weather = data["data"]["current"]["weather"]
    location = data["data"]["location"]

    row = {
        "timestamp": pollution["ts"],
        "city": data["data"]["city"],
        "state": data["data"]["state"],
        "country": data["data"]["country"],
        "lat": location["coordinates"][1],
        "lon": location["coordinates"][0],
        "aqius": pollution["aqius"],
        "mainus": pollution["mainus"],
        "aqicn": pollution["aqicn"],
        "maincn": pollution["maincn"],
        "temperature": weather["tp"],
        "humidity": weather["hu"],
        "pressure": weather["pr"],
        "wind_direction": weather["wd"],
        "wind_speed": weather["ws"]
    }

    file_path = "data/weather_init.csv"
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print("Data written successfully")

else:
    print(f"Error {response.status_code}: {response.text}")
