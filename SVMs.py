import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets

# Dataset co san cua iris =>Demo 
# irisdata = datasets.load_iris()
# X1 = irisdata.data[:, :2]
# y = irisdata.target

# Su dung dataset cua minh
iris = pd.read_csv("iris.csv")
iris = iris.replace(to_replace = "Iris-setosa", value = 0) 
iris =iris.replace(to_replace = "Iris-versicolor", value = 1) 
iris =iris.replace(to_replace = "Iris-virginica", value = 2) 

# print(iris)
X = iris.iloc[:, :2].values
y = iris.iloc[:,-1].values

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# irisdata = pd.read_csv(url, names=colnames)

# X = irisdata.drop('Class', axis=1)
# y = irisdata['Class']

# svm = SVC(C=0.5, kernel='linear')
# svm.fit(X, y)

# plot_decision_regions(X, y, clf=svm, legend=2)

# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.title('SVM on Iris')
# plt.show()

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
# svclassifier = SVC(kernel='poly')
# svclassifier = SVC(kernel='rbf')
# svclassifier = SVC(kernel='sigmoid')

# test chon degree
deg = [0,1,2,3,4,5,6,7,8,9,10]
for i in deg:
    pass
    # svclassifier = SVC(kernel='poly', degree=i)
    # svclassifier.fit(X_train, y_train)
    # plot_decision_regions(X_train, y_train, clf=svclassifier, legend=2)
    # plt.xlabel('sepal length [cm]')
    # plt.ylabel('petal length [cm]')
    # plt.title('SVM on Iris with kernel=' + str(i))
    # plt.show()

svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

# print(X_test.iloc[0])
# Confusiob matrix , theo hang cheo la cac label predict dung
print(confusion_matrix(y_test, y_pred))
# danh gia mo hinh
print(classification_report(y_test, y_pred))

for i in range(len(X_test)):
	print(X_test[i], y_pred[i])

