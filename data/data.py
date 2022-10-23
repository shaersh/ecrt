import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from data.fund_data_names import SECTORS


def get_pcr_data(d=20, a=3, n=5000, test='power'):
    Z_mu = np.zeros((d-1, 1)).ravel()
    Z_Sigma = np.eye(d-1)
    Z = np.random.multivariate_normal(Z_mu, Z_Sigma, n)
    v = np.random.normal(0, 1, (d-1, 1))
    X_mu = Z @ v
    X = np.random.normal(X_mu, 1, (n, 1))
    u = np.random.normal(0, 1, (d-1, 1))
    beta = np.ones((d, 1))
    if test == 'power':
        Y_mu = (Z @ u) ** 2 + a * X
    elif test == 'error':
        Y_mu = (Z @ u) ** 2
        beta[0] = 0
    Y = np.random.normal(Y_mu, 1, (n, 1))
    scaler_Y = StandardScaler().fit(Y)
    Y = scaler_Y.transform(Y)
    X = np.column_stack((X, Z))
    return X, Y, beta


def get_hiv_data(n=1555):
    df = pd.read_csv('../data/HIV.csv').dropna()
    features_names = df.columns[1:]
    n = min(n, df.shape[0])
    X = df.to_numpy()[:n, 1:]
    Y = np.expand_dims(df.to_numpy()[:n, 0], axis=1)
    return X, Y, features_names


def get_hiv_clf(X, j):
    features_idx = np.arange(X.shape[1])
    train_features = np.setdiff1d(features_idx, j)
    train_set = X[:, train_features]
    labels = X[:, j]
    clf = LogisticRegressionCV(cv=10, random_state=0).fit(train_set, labels)
    return clf


def read_log_sector_data(fund="XLK", value='Open', nyears=10, date="2022-09-21"):
    sector = SECTORS[fund]
    xdata = pd.read_csv(f"../data/xdata_{fund}_{sector}_{value}_{nyears}.csv", index_col='Date')
    xdata = xdata.loc[xdata.index <= date]
    X = np.log(xdata.values[1:] / xdata.values[0:-1])
    ydata = pd.read_csv(f"../data/ydata_{fund}_{sector}_{value}_{nyears}.csv", index_col='Date')
    ydata = ydata.loc[ydata.index <= date]
    Y = np.log(ydata.values[1:] / ydata.values[0:-1])
    beta_df = pd.read_csv(f"../data/data_imp_{fund}_{sector}_{value}_{nyears}.csv")
    beta = beta_df["important"].values
    features_names = beta_df["stock"].values
    return X, Y, beta, features_names
