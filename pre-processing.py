#coding=utf-8
import numpy as np
from sklearn import preprocessing

def normalization_demo():
    X_train = np.array([[ 1., -1.,  2.],
                       [ 2.,  0.,  0.],
                       [ 0.,  1., -1.]])

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    print(X_train_minmax)

def standardization_demo():
    X = np.array([[1., -1., 2.],
                 [2., 0., 0.],
                 [0., 1., -1.]])
    X_scaled = preprocessing.scale(X)
    # scaler = preprocessing.StandardScaler().fit(X)
    # scaler.transform(X)
    print(X_scaled)

if(__name__ == "__main__"):
    normalization_demo()
    standardization_demo()