from sklearn import preprocessing
X = [[ 1., -1.,  2],
      [ 2.,  0.,  0.],
      [ 0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')
print(X_normalized)

#scale all values to make them in a certain range, for example [-1, 1]
normalizer = preprocessing.Normalizer().fit(X)  # fit does nothing
X_normalized2 = normalizer.transform(X)
print(X_normalized2)
print(X_normalized.mean())