import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#classifying iris using svm

X = np.loadtxt('iris.txt', delimiter=',', usecols=[0,1,2,3])
y = np.loadtxt('iris.txt', delimiter=',', usecols=[4], dtype=np.str )
m =  X.shape[0]
# X = np.concatenate((np.ones((m,1)), X),axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.svm import SVC
C = 1
svm_model = SVC(kernel= 'linear', C = C).fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
accuracy = svm_model.score(X_test,y_test)
print(accuracy*100)


