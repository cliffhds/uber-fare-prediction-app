"""
Export Models Script
--------------------
This script trains and exports the winning Stacking Generalization Ensemble
(XGBoost + LightGBM + Linear Regression Meta-Learner) to the 'models/' directory
for real-time inference in the Streamlit application.

Usage:
    python export_models.py
"""

import os
import math
import holidays
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def prepare_data():
    print("Loading datasets...")
    csv_file = 'uberData.csv' if os.path.exists('uberData.csv') else ('uber.csv' if os.path.exists('uber.csv') else None)
    if csv_file is None:
        raise FileNotFoundError("Could not find 'uberData.csv' or 'uber.csv'. Please download it from Kaggle.")

    df = pd.read_csv(csv_file, sep=',', engine='python', on_bad_lines='skip')
    weather_df = pd.read_csv("weatherUber.csv")

    print("Cleaning data...")
    df = df.dropna(subset=['dropoff_longitude', 'dropoff_latitude']).copy()
    df = df[(df['fare_amount'] >= 2.50) & (df['fare_amount'] <= 100.0)]
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]

    # Geospatial NYC bounding box
    nyc_mask = (
        (df['pickup_latitude'] >= 40.57) & (df['pickup_latitude'] <= 40.91) &
        (df['pickup_longitude'] >= -74.15) & (df['pickup_longitude'] <= -73.70) &
        (df['dropoff_latitude'] >= 40.57) & (df['dropoff_latitude'] <= 40.91) &
        (df['dropoff_longitude'] >= -74.15) & (df['dropoff_longitude'] <= -73.70)
    )
    df = df[nyc_mask].copy()

    # Feature Engineering
    print("Engineering features...")
    df['trip_distance_km'] = haversine(
        df['pickup_latitude'].values, df['pickup_longitude'].values,
        df['dropoff_latitude'].values, df['dropoff_longitude'].values
    )
    df = df[df['trip_distance_km'] > 0.05].copy()

    # Temporal features
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['pickup_datetime']).copy()

    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['date'] = df['pickup_datetime'].dt.date

    # Merge weather
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE']).dt.date
    weather_sub = weather_df[['DATE', 'PRCP', 'SNOW', 'TMAX', 'AWND']].copy()
    weather_sub['AWND'] = weather_sub['AWND'].fillna(0)
    weather_sub['PRCP'] = weather_sub['PRCP'].fillna(0)
    weather_sub['SNOW'] = weather_sub['SNOW'].fillna(0)
    weather_sub['TMAX'] = weather_sub['TMAX'].fillna(weather_sub['TMAX'].mean())

    df = df.merge(weather_sub, left_on='date', right_on='DATE', how='left')
    df['PRCP'] = df['PRCP'].fillna(0)
    df['SNOW'] = df['SNOW'].fillna(0)
    df['TMAX'] = df['TMAX'].fillna(df['TMAX'].mean() if not df['TMAX'].empty else 60)
    df['AWND'] = df['AWND'].fillna(0)

    # Holidays
    years = df['year'].unique().tolist()
    ny_holidays = holidays.US(state='NY', years=years)
    df['is_holiday'] = df['date'].apply(lambda d: 1 if d in ny_holidays else 0)

    features = [
        'trip_distance_km',
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'passenger_count', 'hour', 'day_of_week', 'month', 'year',
        'PRCP', 'SNOW', 'TMAX', 'AWND', 'is_holiday'
    ]
    target = 'fare_amount'

    return df[features], df[target]

def train_and_export():
    X, y = prepare_data()
    print(f"Dataset ready: {X.shape[0]} rows, {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    print("Training LightGBM Regressor...")
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    xgb_oof_train = np.zeros(X_train.shape[0])
    lgb_oof_train = np.zeros(X_train.shape[0])
    xgb_oof_test = np.zeros(X_test.shape[0])
    lgb_oof_test = np.zeros(X_test.shape[0])

    print("Generating Out-Of-Fold predictions for Stacking...")
    for train_ix, val_ix in kf.split(X_train):
        X_tr, y_tr = X_train.iloc[train_ix], y_train.iloc[train_ix]
        X_va = X_train.iloc[val_ix]

        xgb_model.fit(X_tr, y_tr)
        xgb_oof_train[val_ix] = xgb_model.predict(X_va)
        xgb_oof_test += xgb_model.predict(X_test) / 5.0

        lgb_model.fit(X_tr, y_tr)
        lgb_oof_train[val_ix] = lgb_model.predict(X_va)
        lgb_oof_test += lgb_model.predict(X_test) / 5.0

    print("Fitting Meta-Learner (Linear Regression)...")
    stack_X_train = pd.DataFrame({'xgb': xgb_oof_train, 'lgb': lgb_oof_train})
    stack_X_test = pd.DataFrame({'xgb': xgb_oof_test, 'lgb': lgb_oof_test})

    meta_model = LinearRegression()
    meta_model.fit(stack_X_train, y_train)

    y_pred = meta_model.predict(stack_X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- FINAL STACKING RESULTS ---")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE:  ${mae:.2f}")
    print(f"R2:   {r2:.4f}")

    # Retrain base models on full training data
    print("\nRetraining base learners on full training dataset for export...")
    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Saving model artifacts to {models_dir}...")
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model.joblib"), compress=3)
    joblib.dump(lgb_model, os.path.join(models_dir, "lgb_model.joblib"), compress=3)
    joblib.dump(meta_model, os.path.join(models_dir, "meta_model.joblib"), compress=3)
    print("Done! Models exported successfully.")

if __name__ == '__main__':
    train_and_export()
