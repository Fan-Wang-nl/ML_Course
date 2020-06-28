from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

def dataset_Demo():
    """
    usage of sklearn dataset
    :return:
    """
    #-----------------------------------------------------------------------------------------------
    #load data from internet
    iris = load_iris()
    #查看完整数据集
    print("-----------------------iris datasets:--------------------\n", iris)
    #数据集描述
    print("-----------------------show data set description:----------------------\n", iris["DESCR"])
    #特征的名字
    print("-----------------------feature names:--------------------\n", iris.feature_names)
    #特征值，特征值数据的大小
    print("-----------------------feature value:--------------------\n", iris.data, iris.data.shape)

    # -----------------------------------------------------------------------------------------------
    #split dataset
    #train_test_split(iris, iris.target)        #the default value of test_size is 0.25
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    #训练集的特征值
    print("-------------------training set shape----------------------\n", X_train.shape)

    # -----------------------------------------------------------------------------------------------
    #fetch dataset
    # data_all = fetch_20newsgroups(subset='all', shuffle=True, random_state = 42)
    # print(data_all)
    return None

if __name__ == "__main__":
    #solution 1:usage of sklearn dataset
    dataset_Demo()



