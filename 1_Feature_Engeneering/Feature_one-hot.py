from sklearn.feature_extraction import DictVectorizer
import numpy as np
v = DictVectorizer(dtype=np.float64, sparse=True, sort=True)
D = [{'foo': 1, 'bar': 2}, {'fox': 3, 'baz': 1}]
X = v.fit_transform(D)
print(type(v.fit_transform(D)))
print(v.vocabulary_)
print(v.get_feature_names())
print(X.toarray())
