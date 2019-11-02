# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

def demo_CountVec():
    # 文本文档列表
    text = ["The quick brown fox jumped over the lazy dog."]
    # 构造变换函数
    vectorizer = CountVectorizer()

    # vector = vectorizer.fit_transform(text) inlcudes fit and transform operations

    # 词条化以及建立词汇表
    vectorizer.fit(text)
    # 总结,通过词汇表来查看到底是什么被词条化了
    print("the vocabulary of the text:")
    print(vectorizer.vocabulary_)
    # 编码文档
    vector = vectorizer.transform(text)#vectorize the data
    # 总结编码文档, 可以看到，词汇表中有 8 个单词，于是编码向量的长度为 8。
    print("vector shape:")
    print(vector.shape)
    print("vector type:")
    print(type(vector))
    print(vector)
    print(vector.toarray())


def demo_TfidfVec():
    # 文本文档列表
    text = ["The quick brown fox jumped over the lazy dog.",
            "The dog.",
            "The fox"]
    # 创建变换函数
    vectorizer = TfidfVectorizer()
    # 词条化以及创建词汇表
    vectorizer.fit(text)
    # 总结
    print("the vocabulary of the text:")
    print(vectorizer.vocabulary_)
    print("Inverse document frequency:")
    print(vectorizer.idf_)
    # 编码文档
    print("----------------------------tfidf for data[0]-----------------------------")
    vector = vectorizer.transform([text[0]])
    print(vector.shape)
    print(vector.toarray())

    print("----------------------------tfidf for data[1]-----------------------------")
    vector = vectorizer.transform([text[1]])
    print(vector.shape)
    print(vector.toarray())

    print("----------------------------tfidf for data[2]-----------------------------")
    vector = vectorizer.transform([text[2]])
    print(vector.shape)
    print(vector.toarray())

if(__name__ == "__main__"):
    # 1 count vector for text feature extraction, Bag-of-words model
    demo_CountVec()
    # 2 term frequency vector for text feature extraction
    demo_TfidfVec()
