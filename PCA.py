import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot

def PCA_demo():
    #get a pandas.core.frame.DataFrame instance
    df_wine = pd.read_csv('D:/PythonProjects/wine.csv',header=None)
    print(type(df_wine))
    print(df_wine.head())
    print("shape: ", df_wine.shape)

    #iloc is a method to get a specified area of data
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    print("X", X)#X is data for features
    print("y", y)#y is the output
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    print("type(X_train_std): ", type(X_train_std))
    X_test_std = sc.fit_transform(X_test)

    covariant_matrix = np.cov(X_train_std.T)
    print("covariant_matrix.shape: ", covariant_matrix.shape)

    eigen_values, eigen_vectors = np.linalg.eig(covariant_matrix)
    print("eigen_values: ", eigen_values)
    print("eigen_vectors: ", eigen_vectors[::5])

    tot = sum(eigen_values)
    var_exp = [(i / tot) for i in sorted(eigen_values, reverse=True)]
    cum_var_exp = np.cumsum(var_exp) #Return the cumulative sum of the elements along a given axis.

    pyplot.bar(range(1,14), var_exp, alpha=0.5, align='center',
                      label='individual explained variance')
    pyplot.step(range(1,14), cum_var_exp, where='mid',
                      label='cumulative explained variance')
    pyplot.ylabel('Explained variance ratio')
    pyplot.xlabel('Principal components')
    pyplot.legend(loc='best')
    pyplot.show()

    #get a list of pairs of eigen value and eigen vector
    eigen_pairs = [(np.abs(eigen_values[i]),eigen_vectors[:,i]) for i in range(len(eigen_values))]
    eigen_pairs.sort(reverse=True)
    #horizontal stack. All arrays must have the same shape along all but the second axis.
    w= np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    print("w.shape: ", w.shape, "\nw: ")
    print(w)

    print("X_train_std[0]: ", X_train_std[0])
    X_train_pca0 = X_train_std[0].dot(w)
    print("X_train_pca0: ", X_train_pca0)

    X_train_pca = X_train_std.dot(w)

    #最后我们图示下最终的124个样本在二维空间的分布
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for i, c, m in zip(np.unique(y_train), colors, markers):
        pyplot.scatter(X_train_pca[y_train==i, 0], X_train_pca[y_train==i, 1],
                c=c, label=i, marker=m)
    pyplot.xlabel('PC 1')
    pyplot.ylabel('PC 2')
    pyplot.legend(loc='lower left')
    pyplot.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    from matplotlib.colors import ListedColormap
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    pyplot.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pyplot.xlim(xx1.min(), xx1.max())
    pyplot.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        pyplot.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                marker=markers[idx], label=cl)
def sklearn_PCA_demo():
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA

    # get a pandas.core.frame.DataFrame instance
    df_wine = pd.read_csv('D:/PythonProjects/wine.csv', header=None)
    print(type(df_wine))
    print(df_wine.head())
    print("shape: ", df_wine.shape)

    # iloc is a method to get a specified area of data
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    print("X", X)  # X is data for features
    print("y", y)  # y is the output
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.fit_transform(X_test)

    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    #use linear regression to classify data after PCA
    lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    pyplot.xlabel('PC1')
    pyplot.ylabel('PC2')
    pyplot.legend(loc='lower left')
    pyplot.show()

def sklearn_PCA_demo2():
    from sklearn.decomposition import PCA
    data = [[ 1.  ,  1.  ],
       [ 0.9 ,  0.95],
       [ 1.01,  1.03],
       [ 2.  ,  2.  ],
       [ 2.03,  2.06],
       [ 1.98,  1.89],
       [ 3.  ,  3.  ],
       [ 3.03,  3.05],
       [ 2.89,  3.1 ],
       [ 4.  ,  4.  ],
       [ 4.06,  4.02],
       [ 3.97,  4.01]]
    pca = PCA(n_components=1)
    newData = pca.fit_transform(data)
    print(newData)

def sklearn_PCA_demo3():
    from sklearn.decomposition import PCA
    # get a pandas.core.frame.DataFrame instance
    df_wine = pd.read_csv('D:/PythonProjects/wine.csv', header=None)
    # iloc is a method to get a specified area of data
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    pca = PCA(n_components=12)
    data_new = pca.fit_transform(X)

    print("data_new.shape: ",data_new.shape)
    print(data_new)

if __name__ == "__main__":
    #PCA_demo()
    #sklearn PCA demo
    sklearn_PCA_demo3()