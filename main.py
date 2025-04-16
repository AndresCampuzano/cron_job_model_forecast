import requests
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Constants
CITY_ID = os.getenv("CITY_ID")
API_URL = os.getenv("API_URL")

# 1. Fetch data
response = requests.get(API_URL)
response.raise_for_status()
data = response.json()

# 2. Load into DataFrame
df = pd.DataFrame(data)
df['created_at'] = pd.to_datetime(df['created_at'])

# 3. Prepare data for Prophet
df_temp = df[['created_at', 'temperature']].rename(columns={'created_at': 'ds', 'temperature': 'y'})
df_temp['ds'] = df_temp['ds'].dt.tz_localize(None)  # Remove timezone

df_hum = df[['created_at', 'humidity']].rename(columns={'created_at': 'ds', 'humidity': 'y'})
df_hum['ds'] = df_hum['ds'].dt.tz_localize(None)  # Remove timezone

# 4. Train Prophet models
model_temp = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False)
model_temp.fit(df_temp)

model_hum = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False)
model_hum.fit(df_hum)

# 5. Forecast next 24 hours (1440 minutes)
future_temp = model_temp.make_future_dataframe(periods=1440, freq='min')
future_hum = model_hum.make_future_dataframe(periods=1440, freq='min')

forecast_temp = model_temp.predict(future_temp)
forecast_hum = model_hum.predict(future_hum)

# Get the last timestamp from the real data and make it timezone-naive
last_real_timestamp = df['created_at'].max().tz_localize(None)

# Filter the predicted data to start after the last real timestamp
predicted_temp_after_real = forecast_temp[forecast_temp['ds'] > last_real_timestamp]
predicted_hum_after_real = forecast_hum[forecast_hum['ds'] > last_real_timestamp]

# Calculate hourly averages for the last 5 hours of real data (exclude current hour if incomplete)
last_5_hours_mask = df['created_at'] >= (df['created_at'].max() - pd.Timedelta(hours=5))
last_5_hours_real = (
    df.loc[last_5_hours_mask]
    .set_index('created_at')
    .resample('h')
    .mean(numeric_only=True)
)

# Drop the current hour if it is incomplete
if last_5_hours_real.index[-1] > df['created_at'].max().floor('h'):
    last_5_hours_real = last_5_hours_real.iloc[:-1]

# Calculate hourly averages for the first 5 hours of predicted data
predicted_temp_after_real = predicted_temp_after_real.copy()
predicted_temp_after_real.loc[:, 'hour'] = predicted_temp_after_real['ds'].dt.floor('h')

predicted_hum_after_real = predicted_hum_after_real.copy()
predicted_hum_after_real.loc[:, 'hour'] = predicted_hum_after_real['ds'].dt.floor('h')

hourly_avg_temp_predicted = predicted_temp_after_real.groupby('hour')['yhat'].mean().head(5)
hourly_avg_hum_predicted = predicted_hum_after_real.groupby('hour')['yhat'].mean().head(5)

# Print last 5 entries of real data
print("\nLast 5 entries of real data:")
print(df[['created_at', 'temperature', 'humidity']].tail(5))

# Print first 5 entries of predicted data after the real data
print("\nFirst 5 entries of predicted temperature data after the real data:")
print(predicted_temp_after_real[['ds', 'yhat']].head(5))

print("\nFirst 5 entries of predicted humidity data after the real data:")
print(predicted_hum_after_real[['ds', 'yhat']].head(5))

# Print hourly averages for the last 5 hours of real data
print("\nHourly averages of real data (last 5 hours):")
print(last_5_hours_real[['temperature', 'humidity']])

# Print hourly averages for the first 5 hours of predicted data
print("\nHourly averages of predicted temperature data (first 5 hours):")
print(hourly_avg_temp_predicted)

print("\nHourly averages of predicted humidity data (first 5 hours):")
print(hourly_avg_hum_predicted)

# Optional: show plots
model_temp.plot(forecast_temp)
plt.title("Temperature Forecast (24h)")
plt.grid(True)
plt.show()

model_hum.plot(forecast_hum)
plt.title("Humidity Forecast (24h)")
plt.grid(True)
plt.show()
