# Money Demand Forecasting — Indian Economy

Code for the paper:

**Sikhwal S., Sen S.** "Comparative Analysis of Machine Learning Models for Money Demand Forecasting in the Indian Economy."
*HSE Economic Journal.* 2024; 28(1): 133–158.
DOI: 10.17323/1813-8691-2024-28-1-133-158

## Overview
This repository contains the full Python implementation of ML-based money demand forecasting
for both narrow (M1) and broad (M3) monetary aggregates in India using monthly data (1997–2021).

## Models
- AR(1) — benchmark
- Random Forest, Gradient Boosting, XGBoost
- Support Vector Regression (SVR)
- LASSO
- LSTM (deep learning)

## Validation
Expanding window cross-validation with K = 2 to 7 folds.
Forecast accuracy compared using the Diebold-Mariano test with Harvey adjustment.

## Files
| File | Description |
|------|-------------|
| `M1_clean.ipynb` | Narrow money (M1) forecasting notebook |
| `M3_clean.ipynb` | Broad money (M3) forecasting notebook |
| `helper_functions.py` | Shared utility functions (metrics, DM test, CV splits) |
| `DataMoneyDemandNew.xlsx` | Dataset (CEIC, monthly, India 1997–2021) |

## Requirements
numpy, pandas, matplotlib, scikit-learn, xgboost, statsmodels, arch, scipy, torch
