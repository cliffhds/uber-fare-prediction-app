# 🚕 NYC Uber Fare Prediction & Geospatial Weather Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-11557c?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-2e7d32?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

![Uber NYC Banner](images/uber_banner.png)

An end-to-end Machine Learning and Data Analytics project predicting **Uber ride fares in New York City**. This project integrates over **200,000 transaction records** with high-resolution historical meteorological data from the **National Oceanic and Atmospheric Administration (NOAA)** to uncover the complex non-linear interactions between geospatial distances, temporal rush-hour cycles, and severe weather conditions.

---

## 📌 Executive Summary & Key Highlights

* **Primary Objective:** Build a production-grade predictive regression pipeline estimating NYC Uber ride fares.
* **External Weather Integration:** Joined daily NOAA Central Park climate logs (precipitation, snowfall, wind speed, temperature bounds) to capture demand surges during inclement weather.
* **Geospatial & Temporal Feature Engineering:** Implemented Haversine great-circle distance formulas, NYC bounding-box coordinate filters, diurnal/day-of-week binning, and official US/NY holiday tracking.
* **Winning Model Architecture:** A **Stacked Generalization Ensemble** combining **XGBoost Regressor** (level-wise stability) + **LightGBM Regressor** (leaf-wise sensitivity) through a **Linear Regression Meta-Learner**, achieving an **RMSE of $3.18** and explaining **88.4% of total fare variance** ($R^2 = 0.8836$).
* **Interactive Web App:** Includes a fully functional **Streamlit** dashboard with interactive Folium maps, NYC landmark presets, rush-hour surge diagnostics, and real-time fare predictions.

---

## 📊 Model Benchmark & Comparison

All models were evaluated using **5-Fold Cross-Validation** on unseen test data:

| Model Architecture | RMSE (USD) | $R^2$ Score | Key Characteristics |
| :--- | :---: | :---: | :--- |
| 🏆 **Stacked Generalization (XGB + LGBM + Meta)** | **$3.18** | **0.8836** | **Optimal bias-variance balance; best dollar accuracy** |
| 🥈 **LightGBM Regressor** | $3.20 | 0.8801 | Ultra-fast histogram binning; captures granular weather effects |
| 🥉 **XGBoost Regressor** | $3.23 | 0.8750 | Strong $L_1/L_2$ regularization; robust against coordinate noise |
| 🌲 **Random Forest Regressor (Baseline)** | $3.36 | 0.8210 | Ensemble bagging baseline; higher residual variance |

---

## 📂 Datasets

![NOAA Weather Banner](images/noaa_banner.png)

1. **Uber NYC Fares Dataset:**
   * **Source:** [Kaggle - Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset) (NYC Taxi & Limousine Commission).
   * **Records:** 200,000 trips spanning multiple years.
   * **Fields:** `key`, `fare_amount` (Target), `pickup_datetime`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `passenger_count`.

2. **NOAA Central Park Weather Dataset (`weatherUber.csv`):**
   * **Source:** [NOAA Climate Data Online](https://www.ncdc.noaa.gov/cdo-web/search).
   * **Station:** Central Park, New York City (USW00094728).
   * **Variables:** `PRCP` (Precipitation), `SNOW` (Snowfall), `TMAX` / `TMIN` (Temperatures), `AWND` (Average Wind Speed).

---

## 🛠️ Feature Engineering & Preprocessing Workflow

1. **Outlier Filtering & Quality Control:**
   * Filtered unrealistic fares: bounded between `$2.50` and `$100.00`.
   * Constrained coordinates to Greater NYC bounding coordinates (`40.57 <= lat <= 40.91`, `-74.15 <= lon <= -73.70`), removing ocean and out-of-state GPS anomalies.
   * Dropped rides with 0 or >6 passengers.
2. **Geospatial Metrics:**
   * Computed Haversine great-circle distances in kilometers between pickup and dropoff points.
3. **Temporal Discretization:**
   * Extracted `hour`, `day_of_week`, `month`, `year`, capturing morning/evening rush hours and weekend demand peaks.
4. **Weather Discretization & Holiday Tagging:**
   * Mapped continuous weather records into categorical factors (`clear`, `rainy`, `snowy`).
   * Flagged US and NY state holidays using the Python `holidays` toolkit.

---

## 📁 Repository Structure

```text
├── app.py                    # Interactive Streamlit Web Application
├── export_models.py          # Script to train and serialize Stacking ML models
├── Final File.ipynb          # Clean, fully-executed, documented Jupyter Notebook
├── README.md                 # Project documentation & benchmark overview
├── requirements.txt          # Python environment dependencies
├── .gitignore                # Git ignore configuration
├── weatherUber.csv           # NOAA NYC Central Park meteorological dataset
├── models/                   # Directory containing serialized model artifacts (.joblib)
└── images/
    ├── uber_banner.png       # Banner visualization for Uber NYC
    └── noaa_banner.png       # Banner visualization for NOAA Weather data
```

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/nyc-uber-fare-prediction.git
cd nyc-uber-fare-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```

### 4. (Optional) Train & Export Models
Place `uber.csv` from [Kaggle](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset) in the root folder and run:
```bash
python export_models.py
```

---

## 🌐 Deploy to Streamlit Cloud (1-Click)

1. Push this repository to GitHub.
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New App"** and select:
   * **Repository:** `your-username/nyc-uber-fare-prediction`
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. Click **Deploy!**

---

## 📖 References & Citations

* D. H. Wolpert, *"Stacked Generalization,"* Neural Networks, vol. 5, no. 2, pp. 241-259, 1992.
* T. Chen and C. Guestrin, *"XGBoost: A Scalable Tree Boosting System,"* in *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.
* G. Ke et al., *"LightGBM: A Highly Efficient Gradient Boosting Decision Tree,"* in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
* J. Friedman, *"Greedy Function Approximation: A Gradient Boosting Machine,"* *Annals of Statistics*, vol. 29, pp. 1189-1232, 2001.
