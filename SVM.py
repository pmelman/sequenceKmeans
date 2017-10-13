###########################
## Syntax: python3 SVM.py C(optional, defaults to 0.1)
## Outputs error as percentage
###########################

import sys
import numpy as np
import csv
from sklearn import svm, linear_model, preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import time
# import svmcrossvalidate as cv


def SVM(C=0.1, datafile='temp/kdata', path='temp/'):

	kernel = 'linear'

	data = np.loadtxt(datafile)
	trainlabels = np.loadtxt(path+'trainlabels')
	testlabels = np.loadtxt(path+'testlabels')

	x_train = np.array([data[int(i)] for i in trainlabels[:,1]])
	y_train = np.array(trainlabels[:,0], dtype=np.int32)

	x_test = np.array([data[int(i)] for i in testlabels[:,1]])
	y_test = np.array(testlabels[:,0], dtype=np.int32)

	# if crossval == 'cv':
	# 	C = cv.getbestC(trainData, trainlabels)[0]

	clf = svm.SVC(C=C, kernel = kernel)
	# clf = RandomForestClassifier(min_samples_leaf=20)
	# clf = BaggingClassifier(svm.SVC(C=C, kernel = kernel), n_estimators=10)
	# clf = linear_model.LogisticRegression()
	# clf = linear_model.SGDClassifier(loss='perceptron')

	clf.fit(x_train, y_train)
	x = clf.predict(x_test)
	# print(clf.decision_function(x_test)[0:5], x[0:5])
	error = balancedError(x, y_test)
	auc = roc_auc_score(y_test, clf.decision_function(x_test))
	# auc = 0

	return error, auc

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
	# print("error0:", error0, "error1:", error1)
	error = (error0 + error1) / 2
	return error

# Use -run flag as first argument to run as script
# try:
# 	if sys.argv[1] == '-run':
# 		error, auc = SVM()
# 		print('SVM error:', error)
# 		print("Area under ROC curve:", auc)
# except:
# 	next

if __name__ == "__main__":
	try:
		C = float(sys.argv[1])
		file = sys.argv[2]
	except:
		C = 0.1
		file = 'temp/kdata'

	error, auc = SVM(C, file)
	print('SVM error:', error)
	print("Area under ROC curve:", auc)