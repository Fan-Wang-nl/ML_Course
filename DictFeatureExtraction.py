# coding=utf-8
from sklearn.feature_extraction import DictVectorizer
import numpy as np

def featureExtraction_Demo1():
    v = DictVectorizer(dtype=np.float64, sparse=False, sort=True)
    D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    X = v.fit_transform(D)
    print(type(v.fit_transform(D)))
    print(X)


def featureExtraction_Demo2():
    onehot_encoder = DictVectorizer()
    instances = [{'city': '北京'}, {'city': '天津'}, {'city': '上海'}]
    print(onehot_encoder.fit_transform(instances).toarray())

def featureExtraction_Demo3():
    measurements = [{'city': 'Dubai', 'temperature': 33.},
                    {'city': 'London', 'temperature': 12.},
                    {'city': 'San Fransisco', 'temperature': 18.}]
    vec = DictVectorizer(sparse=True, separator=':', sort=True)
    print(type(vec.fit_transform(measurements)))
    print(vec.fit_transform(measurements).toarray())
    print(vec.feature_names_)
    print(vec.vocabulary_)

if __name__ == "__main__":

    #demo1
    featureExtraction_Demo3()
