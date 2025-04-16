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

# 1. Fetch data
response = requests.get(API_URL)
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

# Print last 5 entries of real data
print("\nLast 5 entries of real data:")
print(df[['hour', 'temperature', 'humidity']].tail(5))

# Print first 5 entries of predicted data after the real data
print("\nFirst 5 entries of predicted temperature data after the real data:")
print(predicted_temp_after_real[['ds', 'yhat']].head(5))

print("\nFirst 5 entries of predicted humidity data after the real data:")
print(predicted_hum_after_real[['ds', 'yhat']].head(5))

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

