from sklearn.feature_selection import VarianceThreshold
"""
VarianceThreshold is a simple baseline approach to feature selection. 
It removes all features whose variance doesn’t meet some threshold. 
By default, it removes all zero-variance features, i.e. features that have the same value in all samples.

As an example, suppose that we have a dataset with boolean features, 
and we want to remove all features that are either one or zero (on or off) in more than 80% of the samples. 
Boolean features are Bernoulli random variables, and the variance of such variables is given by
    p*(1-p)
so we can select using the threshold .8 * (1 - .8):
"""

X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_new = sel.fit_transform(X)
print(X_new)
