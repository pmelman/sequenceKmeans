#########################
## Syntax: python3 runCast.py cast_matrix_filepath number_of_tasks data_filepath
## number_of_tasks is int equal to number of columns in cast matrix file minus one
#########################

import sys
import os
import SVM
import NN
import readData2

try:
	cast = sys.argv[1]
	cols = int(sys.argv[2])
except:
	cast = '../datasets/3PGK/3PGK_15.cast'
	cols = 10

try:
	dataFile = sys.argv[3]
	path = sys.argv[4]
except:
	dataFile = 'temp/kdata'
	path = 'temp/'

auc_sum = 0
err_sum = 0
for i in range(1, cols+1):
	# os.system("python3 readData2.py "+cast+' '+str(i))
	readData2.readData(cast, i, path=path)
	# err, auc = SVM.SVM(datafile=dataFile, path=path, C=.001)
	err, auc = NN.NN(datafile=dataFile, path=path, epochs=10)
	auc_sum += auc
	err_sum += err
	# print('SVM err:', err)
	# print("ROC AUC:", auc)


print("\nAverage AUC:", auc_sum / cols)
print("Average err:", err_sum / cols)