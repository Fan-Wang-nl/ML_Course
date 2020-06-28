"""
Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn;
they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance.
"""
from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
                     [ 2.,  0.,  0.],
                     [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled)

#the purpose of standardization is to make the mean become 0
print(X_scaled.mean())
print(X_scaled.std(axis=0))