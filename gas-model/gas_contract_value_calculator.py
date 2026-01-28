# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 23:09:52 2023

@author: ahmed

Calculates the value of natural gas contracts for trading, accounting for
seasonal fluctuations and physical storage constraints.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# --- 1. Data Processing & Visualisation ---
# Load data
data = pd.read_csv("Nat_Gas.csv")
data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y')

# Use the very first date in data as the anchor (Day 0) to avoid hardcoding "2020-10-31"
start_date = data['Dates'].min()
data['Days_From_Start'] = (data['Dates'] - start_date).dt.days

# Plot initial data
plt.figure(figsize=(10, 6))
plt.plot(data['Dates'], data['Prices'], '-', label='Historical Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Natural Gas Prices (Historical)')
plt.legend()
plt.show()

# --- 2. Model ---
# Fit Trend + Sin + Cos all at once.
# This prevents the seasonal cycle from skewing the linear trend line.

# Feature Engineering
# Time variable for trend
X_days = data['Days_From_Start'].values.reshape(-1, 1)

# Seasonal variables (Annual cycle = 365.25 days)
X_sin = np.sin(2 * np.pi * data['Days_From_Start'] / 365.25).values.reshape(-1, 1)
X_cos = np.cos(2 * np.pi * data['Days_From_Start'] / 365.25).values.reshape(-1, 1)

# Combine into a single matrix for regression
X_final = np.hstack((X_days, X_sin, X_cos))
y = data['Prices'].values

# Fit Model
model = LinearRegression()
model.fit(X_final, y)

print(f"Model Intercept: {model.intercept_:.4f}")
print(f"Model Coefficients: Trend={model.coef_[0]:.5f}, Sin={model.coef_[1]:.4f}, Cos={model.coef_[2]:.4f}")

# --- 3. Price Prediction Function ---

def predict_price(target_date):
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
        
    days = (target_date - start_date).days
    
    # Recreate the features for the new date
    feat_sin = np.sin(2 * np.pi * days / 365.25)
    feat_cos = np.cos(2 * np.pi * days / 365.25)
    
    # Predict
    # Note: model.predict expects a 2D array, so we wrap the features in [[]]
    pred_price = model.predict([[days, feat_sin, feat_cos]])[0]
    return pred_price

# Visualize predictions into the future
future_dates = pd.date_range(start=start_date, end='2025-12-31', freq='D')
future_days = (future_dates - start_date).days

# Vectorised prediction for plotting (faster than a loop)
fut_sin = np.sin(2 * np.pi * future_days / 365.25)
fut_cos = np.cos(2 * np.pi * future_days / 365.25)
X_future = np.column_stack((future_days, fut_sin, fut_cos))
predicted_curve = model.predict(X_future)

plt.figure(figsize=(10, 6))
plt.plot(data['Dates'], y, 'o', label='Observed Data')
plt.plot(future_dates, predicted_curve, '-', label='Model Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Natural Gas Price Forecast (Trend + Seasonality)')
plt.legend()
plt.show()

# --- 4. Contract Valuation Logic ---

def calculate_contract_value(injection_dates, withdrawal_dates, 
                             injection_rate, withdrawal_rate, 
                             max_storage_volume, storage_cost_per_day,
                             fixed_trade_volume=None):
    """
    Calculates the value of a storage contract.
    - Uses the model to predict prices on the transaction dates.
    - Limits trade volume based on injection_rate constraints.
    """
    
    total_value = 0
    
    # Zip allows me to iterate through pairs of injection/withdrawal dates
    for inj_date, with_date in zip(injection_dates, withdrawal_dates):
        
        # 1. Get predicted prices for these dates
        buy_price = predict_price(inj_date)
        sell_price = predict_price(with_date)
        
        # 2. Determine Volume
        # We cannot inject more than the rate allows in a single day.
        # If we assume these are single-day trades, volume is capped by rate.
        max_possible_injection = injection_rate
        max_possible_withdrawal = withdrawal_rate
        
        # If a fixed volume isn't specified, we trade the maximum possible for a single day
        # (capped by storage size and rate limits)
        if fixed_trade_volume:
            volume = min(fixed_trade_volume, max_storage_volume)
        else:
            volume = min(max_storage_volume, max_possible_injection, max_possible_withdrawal)
            
        # 3. Calculate Time held
        storage_days = (with_date - inj_date).days
        if storage_days <= 0:
            print(f"Warning: Withdrawal date {with_date.date()} is before injection date {inj_date.date()}. Skipping.")
            continue
            
        # 4. Calculate Costs and Revenue
        revenue = (sell_price - buy_price) * volume
        cost = storage_cost_per_day * volume * storage_days
        
        profit = revenue - cost
        total_value += profit
        
        print(f"Trade: Inj {inj_date.date()} @ ${buy_price:.2f} -> With {with_date.date()} @ ${sell_price:.2f}")
        print(f"       Volume: {volume} units | Duration: {storage_days} days | Profit: ${profit:,.2f}")

    return total_value

# --- Example Usage ---

injection_dates = [pd.Timestamp('2023-06-15'), pd.Timestamp('2023-08-15')]
withdrawal_dates = [pd.Timestamp('2023-12-15'), pd.Timestamp('2024-02-15')]

# Rates define how fast we can move gas. 
# We cannot move 1,000,000 units in one day if the rate is 50,000.
injection_rate = 50000 
withdrawal_rate = 70000
max_storage_volume = 1000000
storage_cost_per_day = 0.01 # Lowered slightly for a realistic example

contract_val = calculate_contract_value(
    injection_dates, withdrawal_dates,
    injection_rate, withdrawal_rate,
    max_storage_volume, storage_cost_per_day
)

print(f"\nTotal Contract Value: ${contract_val:,.2f}")
