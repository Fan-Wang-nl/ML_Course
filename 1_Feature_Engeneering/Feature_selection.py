#coding=utf-8
import math
import scipy
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def VarianceThreshold_demo():
    X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
    sel = VarianceThreshold(threshold=(0.2))
    y = sel.fit_transform(X)
    print(y)


"""
Pearson corelation coefficient realization
"""
def pearson(vector1, vector2):
    n = len(vector1)
    #simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    #sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    #分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den

def Pearson_demo1():
    vector1 = [1, 2, 3, 5, 8]
    vector2 = [0.11, 0.12, 0.13, 0.15, 0.18]
    r = pearson(vector1, vector2)
    print("vector1", vector1)
    print("vector2", vector2)
    print("corelation coefficient of vector1 and vector2: ", r)

def Pearson_demo2():
    a = [2, 4, 6, 8]
    b = [4, 8, 12, 16]
    c = [-4, -8, -12, -16]
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    print(scipy.stats.pearsonr(a, b) ) # (1.0,0.0)第一个数代表Pearson相关系数
    print(scipy.stats.pearsonr(a, c))  # (-1.0,0.0)

if(__name__ == "__main__"):
    # VarianceThreshold_demo()
    Pearson_demo1()
    Pearson_demo2()