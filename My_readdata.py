#!/usr/bin/env Python
# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler,\
    Normalizer, QuantileTransformer
import os
import re
import scipy.io as sio
from scipy.io import loadmat
import joblib

def read_data(path, step, pathsnames):
    data_dir = path
    data = loadmat(data_dir)
    std = MaxAbsScaler()
    timedata1 = std.fit_transform(data['values'])
    joblib.dump(std, pathsnames)
    lens = len(timedata1)
    X = []
    Y = []
    for i in range(lens-step):
        X.append(timedata1[i:i+step, :])
        Y.append(timedata1[i+step, :])

    return np.asarray(X), np.asarray(Y)

def Read_ALL_Test(path, step, pathsnames):
    data_dir = path
    data = loadmat(data_dir)
    std = MaxAbsScaler()
    timedata1 = std.fit_transform(data['values'])
    joblib.dump(std, pathsnames)
    lens = len(timedata1)
    X = []
    for i in range(lens - step):
        X.append(timedata1[i:i + 1, :])

    return np.asarray(X)


def read_data_fe(path, step, pathsnames):
    data_dir = path
    data = loadmat(data_dir)
    std = MaxAbsScaler()
    timedata1 = std.fit_transform(data['values'])
    joblib.dump(std, pathsnames)

    return timedata1

