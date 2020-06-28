# coding = utf-8
# coding:utf-8  
__author__ = "Wang Fan"
import jieba
# import jieba.posseg as pseg  
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def Tfidf_demo():
    data = [ 'Today the weather is sunny', # 第一类文本切词后的结果，词之间以空格隔开
    'Sunny day weather is suitable to exercise ', # 第二类文本切词后的结果
    'I ate a Hotdog' ] # 第三类文本切词后的结果
    vectorizer = CountVectorizer() # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
    count = vectorizer.fit_transform(data)  # 将文本转为词频矩阵

    print(vectorizer.vocabulary_)
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语 
    print(word)
    print(vectorizer.fit_transform(data))
    print(vectorizer.fit_transform(data).todense())  # 显示词频矩阵

    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值  
    tfidf = transformer.fit_transform(count)  # 计算tf-idf 
    print(tfidf)
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  
    print(weight)

    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
        print
        u"-------这里输出第", i + 1, u"类文本的词语tf-idf权重------"
        for j in range(len(word)):
            print(word[j], weight[i][j])


        TfidfVec = TfidfVectorizer()
        count2 = TfidfVec.fit_transform(data)
        print("--------直接使用TfidfVectorizer()-------")
        print(TfidfVec.fit_transform(data).todense())

def cut_words(text):
    """
    use jieba to seperate Chinese sentences.
    Because in Chinese sentences, there are no blank between words
    :param text:
    :return:
    """
    x = jieba.cut(text)
    print(type(x))
    mylist = list(x)
    text = " ".join(mylist)#joint them together
    return  text;

def Tfidf_CN_demo():
    data = ["我来到北京清华大学",    # 第一类文本切词后的结果，词之间以空格隔开  
    # "他来到了网易杭研大厦",    # 第二类文本的切词结果  
    # "小明硕士毕业与中国科学院",    # 第三类文本的切词结果  
    "我爱北京天安门"]  # 第四类文本的切词结果  
    data_new = []
    for item in data:
        data_new.append(cut_words(item))
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值  
    tfidf = transformer.fit_transform(vectorizer.fit_transform(data_new))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语  
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
        print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            print(word[j], weight[i][j])

if __name__ == "__main__":
    # Tfidf_demo()
    Tfidf_CN_demo()
