from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=3)

dt_model = DecisionTreeClassifier()
dt_model.fit(train_x, train_y)
predict_y = dt_model.predict(test_x)
score = dt_model.score(test_x, test_y)

print(predict_y)
print(test_y)
print('score：', score)