# Machine Learning Models for Money Demand Forecasting — Indian Economy

Code and data for the paper:

**Sikhwal S., Sen S.** "Comparative Analysis of Machine Learning Models for Money Demand Forecasting in the Indian Economy."
*HSE Economic Journal*, 28(1) (2024), 133–158.
DOI: [10.17323/1813-8691-2024-28-1-133-158](http://doi.org/10.17323/1813-8691-2024-28-1-133-158)

---

## Overview

This repository implements a comparative analysis of machine learning models
for forecasting money demand in the Indian economy. Using monthly data from
1997 to 2021, the study forecasts both narrow money (M1) and broad money (M3)
aggregates and compares the predictive performance of six ML models against an
AR(1) benchmark. The Diebold-Mariano test with Harvey adjustment is used to
assess whether differences in forecast accuracy are statistically significant.

Key findings: LSTM outperforms all other models for narrow money (M1),
while LASSO is the best performer for broad money (M3).

---

## Files

| File | Description |
|------|-------------|
| `M1_Forecasting.ipynb` | Forecasting notebook for narrow money (M1) |
| `M3_Forecasting.ipynb` | Forecasting notebook for broad money (M3) |
| `helper_functions.py` | Shared utility functions: metrics, DM test, cross-validation splits |
| `DataMoneyDemand.xlsx` | Raw panel dataset sourced from CEIC |

---

## Data

**Source:** CEIC Data. Monthly frequency, India, January 1997 – December 2021.

Seasonal adjustments applied to M1, M3, IIP, and CPI using the X13-ARIMA method.

### Variables

| Column | Description |
|--------|-------------|
| `M1SA` | Seasonally adjusted nominal narrow money aggregate |
| `M3SA` | Seasonally adjusted nominal broad money aggregate |
| `CPISA` | Seasonally adjusted Consumer Price Index (used to deflate money balances) |
| `IPISA` | Seasonally adjusted Index of Industrial Production (income proxy) |
| `Call money rate` | Short-term interest rate for M1 model (%, not log-transformed) |
| `Govt securities yield` | Interest rate for M3 model (%, not log-transformed) |
| `NEER` | Nominal Effective Exchange Rate |
| `BSE mkt cap mn` | Bombay Stock Exchange market capitalisation (financial stability proxy) |

### Data preparation

Real money balances are computed by deflating the seasonally adjusted nominal
money stock by the seasonally adjusted CPI:

```python
df['M1_real'] = df['M1SA'] / df['CPISA']
df['M3_real'] = df['M3SA'] / df['CPISA']
```

Following the log-linearised Money Demand Function (Equation 1 in the paper),
natural log is applied to all variables except the interest rate variables,
which are in percentage form. First differencing is then applied to achieve
stationarity, confirmed by ADF and Phillips-Perron unit root tests.

---

## Methodology

### Models

| Model | Type | Library |
|-------|------|---------|
| AR(1) | Univariate autoregression — benchmark | `statsmodels` |
| Random Forest | Ensemble (bagging) | `sklearn` |
| Gradient Boosting | Ensemble (boosting) | `sklearn` |
| XGBoost | Regularised boosting | `xgboost` |
| SVR | Support Vector Regression | `sklearn` |
| LASSO | Regularised linear regression (α = 1) | `sklearn` |
| LSTM | Deep learning — recurrent neural network | `torch` |

### Validation

Expanding window cross-validation with K = 2 to 7 folds, maintaining
temporal order (no data leakage). The dataset is split at January 2018:
~80% training (1997–2017), ~20% testing (2018–2021), giving N = 48 test
observations.

### LSTM hyperparameter tuning

Exhaustive grid search over hidden size {32, 64, 128}, number of layers
{1, 2, 3}, dropout rate {0.0, 0.1, 0.2}, and learning rate
{0.0001, 0.0005, 0.0008, 0.003}. Best hyperparameters selected by lowest
validation MSE. Trained for 300 epochs with SGD optimiser.

### Evaluation metrics

MSE, RMSE, MAPE, SMAPE, and Theil Inequality Coefficient (TIC).
Pairwise forecast comparison via the Diebold-Mariano test (Harvey et al.
1997 adjustment) at K = 6.

---

## Requirements

Dependencies: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `statsmodels`,
`arch`, `scipy`, `torch`, `openpyxl`, `jupyter`

---

## How to run

1. Place all files in the same folder
2. Open `M1_Forecasting.ipynb` or `M3_Forecasting.ipynb` in Jupyter or VS Code
3. Run cells sequentially from top to bottom

> **Note:** The LSTM grid search (Section 5.8) is computationally intensive.
> Runtime depends on your hardware — GPU acceleration is used automatically
> if available (`cuda`), otherwise CPU is used.

---

## Citation

```
Sikhwal S., Sen S. (2024). Comparative Analysis of Machine Learning Models
for Money Demand Forecasting in the Indian Economy.
HSE Economic Journal, 28(1), 133–158.
https://doi.org/10.17323/1813-8691-2024-28-1-133-158
```
