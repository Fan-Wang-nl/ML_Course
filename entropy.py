# -*- coding: utf-8 -*-
import math

def entropy(*c):
    print(type(c))
    if (len(c) <= 0):
        return -1
    result = 0
    for x in c:
        result += (-x) * math.log(x, 2)
    return result;

if (__name__ == "__main__"):
    print(entropy(0.99, 0.01));
    print(entropy(0.49, 0.51));
    print(entropy(0.25, 0.25, 0.25, 0.25));
    # x = (0.49, 0.51)
    # print(entropy(list(x)))