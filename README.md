# Weather Forecasting with Prophet

This project uses the [Prophet](https://facebook.github.io/prophet/) library to forecast temperature and humidity for the next 24 hours based on hourly weather data. The forecasts are sent to a server via a POST request.

## Features
- Fetches hourly weather data from an API.
- Uses Prophet to predict temperature and humidity for the next 24 hours.
- Sends predictions to a specified API endpoint.

## Requirements
- Python 3.7 or higher
- See `requirements.txt` for the list of dependencies.

## Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd cron_job_prophet_weather_forecast
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following variables:
   ```
   API_URL=<your_api_url>
   CITY_ID=<your_city_id>
   ```

4. Run the script:
   ```bash
   python main.py
   ```

## Optional
- Uncomment the plotting section in `main.py` to visualize the forecasts.

## License
This project is licensed under the MIT License.
