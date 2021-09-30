import pandas as pd
from sklearn import preprocessing


def min_max_scale(X):
    """
    Scale each column in the provided pd.DataFrame to values between 0 and 1.
    :param X:   (pd.DataFrame)  The data to process.
    :return:    (pd.DataFrame)  The processed data.
    """
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X


def standard_scale(X):
    """
    Standardize each column in the provided pd.DataFrame.
    :param X:   (pd.DataFrame)  The data to process.
    :return:    (pd.DataFrame)  The processed data.
    """
    scaler = preprocessing.StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X
