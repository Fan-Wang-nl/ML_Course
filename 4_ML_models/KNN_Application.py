# coding:utf-8
'''
Created on 2019.05.30
@author:Fan Wang
'''
from sklearn.datasets import load_iris
from sklearn import neighbors
import sklearn

def KNN_iris_demo():
    # 获取与查看iris数据集
    iris = load_iris()
    print("iris: ", iris)
    print("------------------------")


    knn = neighbors.KNeighborsClassifier()

    knn.fit(iris.data, iris.target)
    # 预测
    predict = knn.predict([[0.1, 0.2, 0.3, 0.4]])
    print("predict: ", predict)
    print("------------------------")
    #get the name of the output
    print(iris.target_names[predict])

def KNN_iris_demo2():
    # 1 get data
    iris = load_iris()
    print("iris: ", iris)
    print("------------------------")
    # 2 split data into training set and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, random_state=6)
    print("X_train: ", X_train)
    # 3 feature engineering, standardization
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    # this is very important, because training data and test data should processed by the same method
    X_test = transfer.transform(X_test)
    sample = [[0.1, 0.2, 0.3, 0.4]];
    sample = transfer.transform(sample)
    print("X_train: ", X_train)

    # 4 training process
    knn = neighbors.KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    #5 model assessment
    Y_predict = knn.predict(X_test)
    print("Y_predict: ", Y_predict)
    print("comparision: ", Y_predict == Y_test)

    #5 Accuracy assessment
    score = knn.score(X_test, Y_test)
    print("score: ", score)

    #6 make prediction
    predict = knn.predict(sample)
    print("predict: ", predict)

def KNN_iris_demo3():
    # 获取与查看iris数据集
    iris = load_iris()
    print("iris: ", iris)
    print("------------------------")
    # 划分数据集
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, random_state=6)
    print("X_train: ", X_train)
    # 特征工程，标准化
    # from sklearn.preprocessing import StandardScaler
    # transfer = StandardScaler()
    # X_train = transfer.fit_transform(X_train)
    # # this is very important, because training data and test data should processed by the same method
    # X_test = transfer.transform(X_test)
    # print("X_train: ", X_train)

    # 训练数据集
    knn = neighbors.KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    #model assessment模型评估
    Y_predict = knn.predict(X_test)
    print("Y_predict: ", Y_predict)
    print("comparision: ", Y_predict == Y_test)

    #Accuracy assessment
    score = knn.score(X_test, Y_test)
    print("score: ", score)

    predict = knn.predict([[0.1, 0.2, 0.3, 0.4]])
    print("predict: ", predict)
    print("------------------------")

'''
cross validation and grid search
'''
def KNN_iris_grcv():
    '''
    grid search and cross validation
    :return:
    '''
    # 1 get data
    iris = load_iris()
    print("iris: ", iris)
    print("------------------------")
    # 2 split data into training set and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, random_state=6)
    print("X_train: ", X_train)
    # 3 feature engineering, standardization
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    # this is very important, because training data and test data should processed by the same method
    X_test = transfer.transform(X_test)
    sample = [[0.1, 0.2, 0.3, 0.4]];
    sample = transfer.transform(sample)
    print("X_train: ", X_train)

    # 4 training process
    knn = neighbors.KNeighborsClassifier()

    #bring in grid search and cross validation
    from sklearn.model_selection import  GridSearchCV
    #give serval andidates for parameter k
    param_dict = {"n_neighbors" : [1, 3, 5, 7, 9]}
    #get a new better model
    knn = GridSearchCV(knn, param_grid=param_dict, cv = 10)
    knn.fit(X_train, Y_train)

    #5 model assessment
    Y_predict = knn.predict(X_test)
    print("Y_predict: ", Y_predict)
    print("comparision: ", Y_predict == Y_test)

    #5 Accuracy assessment
    score = knn.score(X_test, Y_test)
    print("score: ", score)

    #6 make prediction
    predict = knn.predict(sample)
    print("predict: ", predict)

def KNN_diabetes_demo():
    import numpy as np
    import pandas as pd
    #get data
    data = pd.read_csv("D:/PythonProjects/pima-indians-diabetes.csv")
    print(data.shape)
    print("data.head(): ", data.head())#first 5 rows
    #split intput and outcome
    X = data.iloc[:, 0:8]
    Y = data.iloc[:, 8]

    #split the training data and test data
    from sklearn.model_selection import train_test_split
    #random_state is a seed which will decide the way of splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=22)

    #prediction
    from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

    model1 = KNeighborsClassifier(n_neighbors=2)
    model1.fit(X_train, Y_train)
    score1 = model1.score(X_test, Y_test)

    model2 = KNeighborsClassifier(n_neighbors=2, weights='distance')
    model2.fit(X_train, Y_train)
    score2 = model2.score(X_test, Y_test)

    model3 = RadiusNeighborsClassifier(n_neighbors=2, radius=500.0)
    model3.fit(X_train, Y_train)
    score3 = model3.score(X_test, Y_test)
    #compare the results of the three models
    print(score1, score2, score3)

    #cross validation
    from sklearn.model_selection import cross_val_score

    result1 = cross_val_score(model1, X, Y, cv=10)
    result2 = cross_val_score(model2, X, Y, cv=10)
    result3 = cross_val_score(model3, X, Y, cv=10)

    print(result1.mean(), result2.mean(), result3.mean())

if(__name__ == "__main__"):
    # KNN_iris_demo()
    # KNN_diabetes_demo()
    # KNN_iris_demo2()
    # KNN_iris_demo3()

    KNN_iris_grcv()