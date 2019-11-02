# coding=utf-8
from math import log
import operator
import sys

import treePlotter

def calcShannonEnt(dataSet):
    """
    计算香农信息增益
    :param dataSet:输入的数据集
    :return: 熵
    """
    numEntries = len(dataSet)  # 数据集实例总数
    labelCounts = {}  # 数据字典，键值是最后一列的数值，记录当前类别出现的次数
    for featVec in dataSet:  # 对于每个数据进行循环
        currentLabel = featVec[-1]  # 最后一列
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1  # 统计这个标签出现的次数
    shannonEnt = 0.0  # 香农信息增益
    for key in labelCounts:  # 对于每个标签
        prob = float(labelCounts[key]) / numEntries  # 获取标签出现的概率
        shannonEnt -= prob * log(prob, 2)  # 信息增益-=xi出现的概率*log2(xi出现的概率)
    return shannonEnt


def createDataSet():
    """
    创造数据集
    :return:数据集，标签
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    :param dataSet:带划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:  # 遍历数据集中的每一组数据
        if featVec[axis] == value:  # 该组数据符合特征
            reducedFeatVec = featVec[:axis]  # 截取该组数据的前半段
            reducedFeatVec.extend(featVec[axis + 1:])  # 截取数据的后半段
            # 这样两次操作删除了以axis为下标的元素
            # 不能直接删除，否则影响原始dataSet
            retDataSet.append(reducedFeatVec)  # 返回的数据集添加上满足条件的数据组去除了特征的数据组
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 最后一列是标签，不是特征
    baseEntropy = calcShannonEnt(dataSet)  # 计算原始香农增益
    bestInfoGain = 0.0  # 最佳信息增益
    bestFeature = -1  # 最好的特征
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        print("uniqueVals", uniqueVals)
        newEntropy = 0.0  # 对于此特征的熵
        for value in uniqueVals:  # 遍历此特征所有的唯一属性值
            print("value", value)
            subDataSet = splitDataSet(dataSet, i, value)  # 按照这个唯一属性值划分数据
            print("subDataSet", subDataSet)
            prob = len(subDataSet) / float(len(dataSet))  # 这个唯一属性值出现的概率
            print("prob", prob)
            newEntropy += prob * calcShannonEnt(subDataSet)  # 对所有唯一属性值得到的熵求和
            print("newEntropy", newEntropy)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        print("infoGain", infoGain)
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            print("bestInfoGain", bestInfoGain)
            bestFeature = i
    return bestFeature  # returns an integer


def majorityCnt(classList):
    """
    如果所有属性都参与了划分，但类标签依然不是唯一的，定义叶子节点的方法
    :param classList: 叶子节点的所有标签
    :return: 该叶子节点的标签定义
    """
    classCount = {}  # 叶子节点的统计
    for vote in classList:  # 投票表决
        if vote not in classCount.keys(): classCount[vote] = 0  # 如果没有该类标签就初始化为0
        classCount[vote] += 1  # 类标签个数加一
    # 也可以用下面代码代替上面两行
    # classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print("sortedClassCount", sortedClassCount)
    # 按照类标签个数排序
    return sortedClassCount[0][0]  # 返回个数最多的标签名称


def createTree(dataSet, labels):
    """
    创建树
    :param dataSet: 数据集
    :param labels: 标签列表，其实用不到
    :return:
    """
    classList = [example[-1] for example in dataSet]  # 所有类别标签
    print("classList", classList)
    if classList.count(classList[0]) == len(classList):  # 判断类标签全部相同
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)  # 已无法再使用特征分类，用标签的大多数代表这个节点
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最佳分类标签的序号
    print("bestFeat", bestFeat)
    bestFeatLabel = labels[bestFeat]  # 最佳分类标签
    print("bestFeatLabel", bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 保存树的所有信息
    del (labels[bestFeat])  # 删除标签列表中的最佳标签
    featValues = [example[bestFeat] for example in dataSet]  # 最佳标签对应的所有特征值
    print("featValues", featValues)
    uniqueVals = set(featValues)  # 把最佳标签对应的所有特征值去重
    print("uniqueVals", uniqueVals)
    for value in uniqueVals:  # 对于每个唯一的最佳标签对应的所有特征值
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    使用决策树的分类函数
    :param inputTree:输入的树
    :param featLabels:特征标签
    :param testVec:要进行分类的向量
    :return:
    """
    firstStr = list(inputTree.keys())[0]  # 输入树的第一个分类标签字符串
    print("firstStr", firstStr)
    secondDict = inputTree[firstStr]  # 标签字符串指向的树
    print("secondDict", secondDict)
    featIndex = featLabels.index(firstStr)  # 将标签字符串转换为索引
    print("featIndex", featIndex)
    key = testVec[featIndex]  # 找出测试的向量此索引下的值
    print("key", key)
    valueOfFeat = secondDict[key]  # 根据索引下的值找出下一个子树
    print("valueOfFeat", valueOfFeat)
    if isinstance(valueOfFeat, dict):  # 循环判断是否已经到了叶节点
        classLabel = classify(valueOfFeat, featLabels, testVec)  # 不是叶子节点，分类标签继续循环
    else:
        classLabel = valueOfFeat  # 已经到了叶节点
    return classLabel  # 返回最后预测的分类标签


def storeTree(inputTree, filename):
    """
    存储决策树
    :param inputTree:要保存的决策树
    :param filename:保存的文件名
    :return:
    """
    import pickle
    fw = open(filename, 'wb')  # 文件写
    pickle.dump(inputTree, fw)  # 把决策树对象序列化写
    fw.close()  # 关闭文件操作


def grabTree(filename):
    """
    从磁盘上读取决策树
    :param filename:文件名字
    :return: 决策树
    """
    import pickle
    fr = open(filename)
    return pickle.load(fr)


dataSet, labels = createDataSet()
print("dataSet", dataSet)
myTree = treePlotter.retrieveTree(0)
print("myTree", myTree)
treePlotter.createPlot(myTree)
print(classify(myTree, labels, [1, 0]))
storeTree(myTree, 'classifierStorage.txt')
print(grabTree('classifierStorage.txt'))