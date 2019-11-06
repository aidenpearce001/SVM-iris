import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

irisdata = pd.read_csv(url, names=colnames)

X = irisdata.drop('Class', axis=1)
y = irisdata['Class']
print(X)
print(y)

# iris = pd.read_csv("iris.csv")
# link iloc va loc : https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
# Lay ra 4 cot dau tien (features cua hoa dien vi) 
# X = iris.iloc[:, 0:4]
# X = iris.drop(iris.columns[4], axis=1)
# print(X)
# Y la class (ten loai hoa dien vi)
# y = (iris.iloc[:,-1])
# print(y)

# Chia data bao gom ca class 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  #0.20 => chia data (80 - train):(20 - test)

# su dung Polynomial Kernel ngoai ra con RBF va sigmoid
svclassifier = SVC(kernel='poly', degree=8 , gamma='auto_deprecated')
# svclassifier = SVC(kernel='rbf')
# svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(X_test.iloc[0])
# print(X_test.iloc[0])
# Confusiob matrix , theo hang cheo la cac label predict dung
print(confusion_matrix(y_test, y_pred))
# danh gia mo hinh
print(classification_report(y_test, y_pred))

for i in range(len(X_test)):
	print( (X_test.iloc[i], y_pred[i]))

