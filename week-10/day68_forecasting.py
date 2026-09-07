"""
Day 68: Time Series Analysis & Forecasting (ARIMA & Moving Averages)
Topic: Time Series, Moving Averages (SMA/EMA), Stationarity (ADF Test), ARIMA Model
Description:
    This exercise covers introductory time series analysis and forecasting:
    - Synthetic time series with linear trend, seasonal component, and noise
    - Calculating Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
    - Stationarity testing with Augmented Dickey-Fuller (ADF) test
    - Fitting an ARIMA(1,1,1) model using statsmodels and forecasting future steps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

plt.switch_backend('Agg')

def generate_time_series(n_periods=60, random_state=42):
    np.random.seed(random_state)
    time = np.arange(n_periods)
    trend = 0.8 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 12)
    noise = np.random.normal(0, 3, size=n_periods)
    series = 50 + trend + seasonality + noise
    
    dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='ME')
    return pd.Series(series, index=dates, name='Sales')

def main():
    print("=" * 70)
    print("DAY 68: Time Series Analysis & ARIMA Forecasting")
    print("=" * 70)

    # 1. Generate Synthetic Monthly Sales Series
    ts = generate_time_series(60)
    print(f"Time Series generated: {len(ts)} monthly observations.")
    print(f"Date range: {ts.index[0].strftime('%Y-%m')} to {ts.index[-1].strftime('%Y-%m')}")

    # 2. Compute Moving Averages
    sma_3 = ts.rolling(window=3).mean()
    ema_3 = ts.ewm(span=3, adjust=False).mean()

    # 3. Stationarity Test (Augmented Dickey-Fuller)
    adf_result = adfuller(ts)
    print("\n--- Augmented Dickey-Fuller (ADF) Test ---")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print("Stationary?" if adf_result[1] < 0.05 else "Non-Stationary (Needs Differencing).")

    # 4. Fit ARIMA(1, 1, 1) Model
    print("\nFitting ARIMA(1, 1, 1) Model...")
    model = ARIMA(ts, order=(1, 1, 1))
    model_fit = model.fit()

    print(model_fit.summary().tables[1])

    # 5. Out-of-Sample Forecast for next 12 months
    forecast_steps = 12
    forecast_obj = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    print(f"\n--- 12-Month Sales Forecast ---")
    print(forecast_mean.round(2))

    # Plot historical series & forecast
    plt.figure(figsize=(12, 5))
    plt.plot(ts.index, ts, label='Historical Sales', color='blue', lw=2)
    plt.plot(sma_3.index, sma_3, label='3-Month SMA', color='orange', linestyle='--')
    plt.plot(forecast_mean.index, forecast_mean, label='ARIMA Forecast', color='red', lw=2)
    plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='95% Confidence Interval')

    plt.title('Time Series Sales Forecasting with ARIMA(1,1,1)')
    plt.xlabel('Date')
    plt.ylabel('Sales Volume')
    plt.legend()

    plt.tight_layout()
    output_png = "ml-exercise/week-10/day68_forecasting.png"
    plt.savefig(output_png)
    plt.close()
    print(f"\nTime series forecast plot saved to '{output_png}'.")

if __name__ == '__main__':
    main()
