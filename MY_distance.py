#!/usr/bin/env Python
# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import os
from scipy.spatial import distance

from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# smooth
def five_point_smoothing(data):
    x = np.arange(len(data))
    f = interp1d(x, data, kind='cubic')
    smoothed_data = f(x)
    return smoothed_data


def is_anomaly(data_point, mean, std, sigma=6):
    return abs(data_point - mean) > sigma * std


def Distance(Data_S, sigma=6):
    MY_features = []
    Distance = []
    Lock = True
    Judge = []
    Degenerate = 0
    Size = len(Data_S)
    lastdata = []
    for i in range(0, int(Size-1)):

        ydata1 = Data_S[i]
        ydata2 = Data_S[i+1]

        ydata1 = five_point_smoothing(ydata1)
        ydata2 = five_point_smoothing(ydata2)

        MY_features.append(ydata1)
        lastdata = ydata2
        if Lock == False:
            x = Judge
        else:
            x = ydata1
        y = ydata2

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        # Calculate Euclidean distance
        euclidean_distances = distance.cdist(x, y, 'euclidean')
        # Calculate Manhattan distance
        manhattan_distances = distance.cdist(x, y, 'cityblock')
        # Calculate Chebyshev distance
        chebyshev_distances = distance.cdist(x, y, 'chebyshev')
        # Calculate Cosine distance
        cosine_distances = distance.cdist(x, y, 'cosine')
        # Calculate Minkowski distance (p=3)
        minkowski_distances = distance.cdist(x, y, 'minkowski', p=3)
        # Dynamic Time Warping (DTW) distance
        dtw_distance, path = fastdtw(x, y, dist=euclidean)

        # Euclidean distance
        euclidean_distance = euclidean_distances[0][0]
        # Manhattan distance
        manhattan_distance = manhattan_distances[0][0]
        # Correlation matrix distance
        corr_matrix = np.corrcoef(x, y)

        distances = pdist(corr_matrix)

        # Calculate Mahalanobis distance
        # Calculate the mean and standard deviation of the two sets of data
        mean1 = np.mean(x)
        mean2 = np.mean(y)
        std1 = np.std(x)
        std2 = np.std(y)
        # Calculate the standardized Euclidean distance as the Mahalanobis distance
        m_distance = np.abs((mean1 - mean2) / np.sqrt(std1 ** 2 + std2 ** 2))

        # Cosine distance
        cosine_distance = cosine_distances[0][0]

        # Calculate Wasserstein distance
        Was_distance = chebyshev_distances[0][0]
        # Cross-entropy
        loss1 = dtw_distance

        Distance.append(np.asarray([euclidean_distance, manhattan_distance, distances[0],
                                    cosine_distance, Was_distance, loss1, m_distance]))
        ALLs = np.asarray(Distance).reshape(len(Distance), -1)

        if Lock == True:
            Judge.append(x)

        if i > 3:

            PasT = ALLs[0:len(ALLs) - 1, :]
            num_elements = len(ALLs) - 1
            selected_indices = np.random.choice(PasT.shape[0], num_elements, replace=False)
            PasT = PasT[selected_indices]
            Nows = ALLs[len(ALLs) - 1, :]
            PasT1 = np.sort(PasT, axis=0)
            median = np.median(PasT1, axis=0)
            mad = np.median(np.abs(PasT1 - median), axis=0)
            madn = mad / 0.675

            My_judge = []

            for k in range(len(Nows)):
                My_judge.append(is_anomaly(Nows[k], median[k], madn[k], sigma=sigma))
                
            # Get the first degradation point
            if sum(np.asarray(My_judge)) >= 5 and Lock and i > 3:

                Judge = np.asarray(Judge)
                Judge = np.mean(Judge[3:, ], axis=0)
                Lock = False
                Degenerate = i

                print('Point {} is the initial degradation point'.format(Degenerate))

    return np.asarray(Distance), Degenerate
