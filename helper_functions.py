import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
import arch.unitroot as at
from scipy.stats import t
import collections


# ── Evaluation metrics ────────────────────────────────────────────────────────

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return np.mean(2 * numerator / denominator) * 100


def theil_inequality_coeff(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return numerator / (np.sqrt(np.mean(np.square(y_true))) + np.sqrt(np.mean(np.square(y_pred))))


def metrics(y_test, y_pred):
    """Compute all five evaluation metrics and return as single-element lists."""
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_test, y_pred)
    tic = theil_inequality_coeff(y_test, y_pred)
    return [np.sqrt(mse)], [mse], [mape], [smape], [tic]


def print_metrics(mean_rmse, mean_mse, mean_mape, mean_smape, mean_tic, split):
    """Print evaluation metrics for a given fold K."""
    print(f"Test Set for K = {split}")
    print(f"Mean of MSE   : {np.mean(mean_mse):.5f}")
    print(f"Mean of RMSE  : {np.mean(mean_rmse):.5f}")
    print(f"Mean of MAPE  : {np.mean(mean_mape):.5f}")
    print(f"Mean of SMAPE : {np.mean(mean_smape):.5f}")
    print(f"Mean of TIC   : {np.mean(mean_tic):.5f}")
    print(" ")


# ── Data splitting ────────────────────────────────────────────────────────────

def make_split(dataframe, X, y, start_date):
    """Split X and y into train/test at start_date."""
    idx = dataframe.index.get_loc(start_date)
    return X[:idx], X[idx:], y[:idx], y[idx:], idx


def AR_expanding_window_cv_with_splits(X, y, n_splits=5):
    """Generate expanding window train/validation splits for use with AR models.

    Unlike expanding_window_cv_with_splits, this function does not fit a model
    internally. Instead it returns a list of (X_train, y_train, X_val, y_val)
    tuples so that the caller can fit any model (e.g. AutoReg) on each fold.

    Args:
        X: feature array (used only for split indexing)
        y: target array
        n_splits: number of expanding window folds (K=2 to 7 in the paper)

    Returns:
        List of (X_train, y_train, X_val, y_val) tuples, one per fold
    """
    n = len(X)
    split_size = n // n_splits
    splits = []
    for i in range(1, n_splits + 1):
        split_end = min(i * split_size, n)
        splits.append((
            X[:split_end],
            y[:split_end],
            X[split_end - 1:split_end],
            y[split_end - 1:split_end],
        ))
    return splits


def expanding_window_cv_with_splits(X, y, model, num_splits=5):
    """Train a sklearn model using expanding window cross-validation.

    For each fold, the model is trained on an incrementally growing training
    window. The model fitted on the final fold is returned for test prediction.

    Args:
        X: feature array
        y: target array
        model: instantiated sklearn-compatible model
        num_splits: number of expanding window folds (K=2 to 7 in the paper)

    Returns:
        y_pred_all: list of validation predictions across all folds
        model: model fitted on the final fold
    """
    n = len(X)
    split_size = n // num_splits
    y_pred_all = []
    for i in range(1, num_splits + 1):
        split_end = min(i * split_size, n)
        X_train, y_train = X[:split_end], y[:split_end]
        X_val = X[split_end - 1:split_end]
        model.fit(X_train, y_train)
        y_pred_all.extend(model.predict(X_val))
    return y_pred_all, model


# ── Stationarity tests ────────────────────────────────────────────────────────

def adf_test(dataframe):
    """Run ADF and Phillips-Perron unit root tests on all columns.

    Null hypothesis: the series has a unit root (non-stationary).
    Maximum lag length: 15 (as per the paper).
    """
    for col in dataframe.columns:
        result = adfuller(dataframe[col].values, autolag='AIC', maxlag=15)
        print(f"{col}:")
        print(f"  ADF Statistic : {result[0]:.4f}")
        print(f"  p-value       : {result[1]:.4f}")
        for key, value in result[4].items():
            print(f"  Critical Value ({key}): {value:.4f}")
        print(f"  Phillips-Perron: {at.PhillipsPerron(dataframe[col].values)}")
        print()


# ── Forecast accuracy test ────────────────────────────────────────────────────

def dm_test(real_values, pred1, pred2, h=1, harvey_adj=True):
    """Diebold-Mariano test for comparing forecast accuracy.

    Compares two forecasts using MSE as the loss criterion.
    Optionally applies the Harvey et al. (1997) small-sample correction.

    Args:
        real_values: actual observed values
        pred1: forecasts from model 1
        pred2: forecasts from model 2
        h: forecast horizon (1 for one-step-ahead)
        harvey_adj: if True, apply Harvey small-sample adjustment

    Returns:
        Named tuple with fields DM (test statistic) and p_value
    """
    real_values = pd.Series(real_values).apply(float).tolist()
    pred1 = pd.Series(pred1).apply(float).tolist()
    pred2 = pd.Series(pred2).apply(float).tolist()

    T = float(len(real_values))

    e1 = [(r - p1) ** 2 for r, p1 in zip(real_values, pred1)]
    e2 = [(r - p2) ** 2 for r, p2 in zip(real_values, pred2)]
    d  = [a - b for a, b in zip(e1, e2)]

    mean_d = pd.Series(d).mean()

    def autocovariance(Xi, N, k, Xs):
        return (1 / float(N)) * sum(
            (Xi[i + k] - Xs) * (Xi[i] - Xs) for i in range(N - k))

    gamma = [autocovariance(d, len(d), lag, mean_d) for lag in range(h)]
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / T

    DM_stat = V_d ** (-0.5) * mean_d

    if harvey_adj:
        DM_stat *= ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** 0.5

    p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    return dm_return(DM=DM_stat, p_value=p_value)
