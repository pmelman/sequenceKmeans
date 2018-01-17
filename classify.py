import sys
import numpy as np
from sklearn import svm, linear_model, preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


class classifier(object):

	def __init__(self, dataFile, castFile, col):

		self.x_train, self.y_train, self.x_test, self.y_test = readData(dataFile, castFile, col)

	def SVM(self, C=0.1, kernel = 'linear'):

		clf = svm.SVC(C=C, kernel = kernel)
		clf.fit(self.x_train, self.y_train)

		x = clf.predict(self.x_test)
		error = balancedError(x, self.y_test)		
		auc = roc_auc_score(self.y_test, clf.decision_function(self.x_test))

		return error, auc

	def logReg(self):

		clf = linear_model.LogisticRegression()
		
		clf.fit(self.x_train, self.y_train)

		x = clf.predict(self.x_test)

		error = balancedError(x, self.y_test)
		auc = roc_auc_score(self.y_test, clf.decision_function(self.x_test))

		return error, auc

	def SGD(self):

		clf = clf = linear_model.SGDClassifier(loss='perceptron')

		clf.fit(self.x_train, self.y_train)

		x = clf.predict(self.x_test)

		error = balancedError(x, self.y_test)
		auc = roc_auc_score(self.y_test, clf.decision_function(self.x_test))

		return error, auc

	
def readData(dataFile, castFile, col):

	data = np.loadtxt(dataFile)
	cast = [l.rstrip().split() for l in open(castFile, 'r').readlines()]

	train = []
	test = []
	trainlabels = []
	testlabels = []

	for i, l in enumerate(cast[1:]):
		if l[col] == '1':
			train.append(data[i])
			trainlabels.append(1)
		if l[col] == '2':
			train.append(data[i])
			trainlabels.append(0)
		if l[col] == '3':
			test.append(data[i])
			testlabels.append(1)
		if l[col] == '4':
			test.append(data[i])
			testlabels.append(0)

	x_train = np.array(train)
	y_train = np.array(trainlabels)

	x_test = np.array(test)
	y_test = np.array(testlabels)

	return x_train, y_train, x_test, y_test

def balancedError(x, y_test):
	# compute balanced error
	error0 = 0
	error1 = 0
	n0 = 0
	n1 = 0

	for i, y in enumerate(y_test):
		if y == 0:
			n0 += 1
			if x[i] != 0:
				error0 += 1
		if y == 1:
			n1 += 1
			if x[i] != 1:
				error1 +=1

	if n0 == 0:
		n0 = 1
	if n1 == 0:
		n1 = 1

	error0 = error0/n0
	error1 = error1/n1
	# print("class 0 error:", error0, "\nclass 1 error:", error1)
	error = (error0 + error1) / 2
	return error
