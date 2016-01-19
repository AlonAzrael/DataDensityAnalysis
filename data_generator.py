

import numpy as np
from numpy import random

import pandas as pd

from sklearn.datasets import make_blobs


def gen_test_data(n_features=3):
    tX, tY = make_blobs(n_samples=100, n_features=n_features, centers=3, cluster_std=1.0, center_box=(1.0, 10.0), shuffle=True)
    return tX, tY


def gen_test_df(n_features=3):
    tX, tY = gen_test_data(n_features=n_features)
    df = pd.DataFrame(tX)

    return df


