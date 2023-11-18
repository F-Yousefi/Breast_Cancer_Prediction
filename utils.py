from sklearn.ensemble import IsolationForest
from scipy.stats import boxcox
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def remove_outliers(x_train, y_train):
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(x_train)
    mask = yhat != -1
    return (x_train[mask], y_train[mask])


def transform_to_norm_dist(x_train):
    all_powers = [-1, -0.5, 0, 0.5, 1]
    for f in x_train.columns:
        ndl = []
        for bcv in all_powers:
            ndf = pd.DataFrame(boxcox(x_train[f] + 0.0001, bcv))
            ndl.append({"skewness": abs(ndf.skew()[0]), "bcv": bcv})
        ndl = pd.DataFrame(ndl)
        x_train[f][:] = boxcox(x_train[f] + 0.0001,
                               ndl["bcv"][np.argmin(ndl['skewness'])])
    return x_train


def maxmin_norm(x_train):
    scaler = MinMaxScaler()
    x_train[:] = scaler.fit_transform(x_train)
    return x_train


def test_train_spliter(
    x_train,
    y_train,
    test_size=0.2): return train_test_split(
        x_train,
        y_train,
        test_size=0.2,
    random_state=40)
