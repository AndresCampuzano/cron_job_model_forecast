import requests
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Constants
API_URL = os.getenv("API_URL")
CITY_ID = os.getenv("CITY_ID")

# 1. Fetch data
response = requests.get(f"{API_URL}/api/weather?city_id={CITY_ID}&hourly_average=true")
response.raise_for_status()
data = response.json()

# 2. Load into DataFrame
df = pd.DataFrame(data)
df['hour'] = pd.to_datetime(df['hour']).dt.tz_localize(None)  # Convert 'hour' to timezone-naive datetime

# 3. Prepare data for Prophet
df_temp = df[['hour', 'temperature']].rename(columns={'hour': 'ds', 'temperature': 'y'})
df_hum = df[['hour', 'humidity']].rename(columns={'hour': 'ds', 'humidity': 'y'})

# 4. Train Prophet models
model_temp = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False)
model_temp.fit(df_temp)

model_hum = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False)
model_hum.fit(df_hum)

# 5. Forecast next 24 hours (hourly intervals)
future_temp = model_temp.make_future_dataframe(periods=24, freq='h')
future_hum = model_hum.make_future_dataframe(periods=24, freq='h')

forecast_temp = model_temp.predict(future_temp)
forecast_hum = model_hum.predict(future_hum)

# Get the last timestamp from the real data
last_real_timestamp = df['hour'].max()

# Filter the predicted data to start after the last real timestamp
predicted_temp_after_real = forecast_temp[forecast_temp['ds'] > last_real_timestamp]
predicted_hum_after_real = forecast_hum[forecast_hum['ds'] > last_real_timestamp]

# Clamp predicted humidity values between 0 and 99
predicted_hum_after_real['yhat'] = predicted_hum_after_real['yhat'].clip(lower=0, upper=99)

# Print last 5 entries of real data
print("\nLast 5 entries of real data:")
print(df[['hour', 'temperature', 'humidity']].tail(5))

# Print first 5 entries of predicted data after the real data
print("\nFirst 5 entries of predicted temperature data after the real data:")
print(predicted_temp_after_real[['ds', 'yhat']].head(5))

print("\nFirst 5 entries of predicted humidity data after the real data:")
print(predicted_hum_after_real[['ds', 'yhat']].head(5))

# 6. Prepare data for POST request
predictions = []
for temp, hum in zip(predicted_temp_after_real.itertuples(), predicted_hum_after_real.itertuples()):
    predictions.append({
        "city_id": CITY_ID,
        "temperature": round(temp.yhat, 1),
        "humidity": round(hum.yhat, 1),
        "forecast_for": temp.ds.strftime("%Y-%m-%dT%H:%M:%SZ")
    })

# 7. Send POST request
post_url = f"{API_URL}/api/predictions"
response = requests.post(post_url, json=predictions)
if response.status_code == 200:
    print("\nPredictions successfully sent to the server.")
else:
    print(f"\nFailed to send predictions. Status code: {response.status_code}, Response: {response.text}")

# Optional: show plots
# model_temp.plot(forecast_temp)
# plt.title("Temperature Forecast (24h)")
# plt.grid(True)
# plt.show()
#
# model_hum.plot(forecast_hum)
# plt.title("Humidity Forecast (24h)")
# plt.grid(True)
# plt.show()

