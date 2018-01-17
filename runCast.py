#########################
## Syntax: python3  runCast.py  cast_matrix_path  data_path  outfile
#########################

import sys, os, time, multiprocessing
import classify

start_time = time.time()

def runData(col):
	global auc_sum
	global err_sum

	clf = classify.classifier(dataFile, castFile, col)
	err, auc = clf.SVM(C=.001)
	# err, auc = clf.SGD()

	return auc, err


try:
	castFile = sys.argv[1]
	dataFile = sys.argv[2]
except:
	castFile = 'sample_data/3PGK_15.cast'
	dataFile = 'temp/kdata'

try:
	outFile = sys.argv[3]
except:
	outFile = 'output'

castID = os.path.split(castFile)[-1][:-5]

castMatrix = [l.rstrip().split('\t') for l in open(castFile, 'r').readlines()]
cols = len(castMatrix[0]) - 1
print(castID)
print(cols, "cols")

pool = multiprocessing.Pool(4)
results = pool.map(runData, [c for c in range(1, cols+1)])
pool.close()

for r in results:
	print("ROC AUC:", r[0])
	# print('clf err:', r[1])

auc_sum = sum([row[0] for row in results])
err_sum = sum([row[1] for row in results])

print("Average AUC:", auc_sum / cols)
print("Average err:", err_sum / cols)

# f = open(outFile, 'w').close() # use this line to reset outFile
f = open(outFile, 'a')
f.write('\n'+castID+'\n')
f.write('Average AUC: '+str(auc_sum / cols)+'\n')
f.write('Average err:'+str(err_sum / cols)+'\n')
f.close()

print("--- done %.2f seconds ---" % (time.time() - start_time))
