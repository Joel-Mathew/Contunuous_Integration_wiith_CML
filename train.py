from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

iris = load_iris()
X = iris.data
y = iris.target

model_dir = 'model'

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

model = LogisticRegression()
model.fit(X_train, y_train)

os.makedirs('model', exist_ok = True)
filename = os.path.join('model','LRmodel.joblib')
joblib.dump(model, filename)