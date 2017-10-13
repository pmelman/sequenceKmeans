import os
import sys
import numpy as np

def readData(castFile, col, path='temp/'):

	# try:
	# 	castFile = open(sys.argv[1], 'r')
	# 	col = int(sys.argv[2])
	# except:
	# 	castFile = open('../datasets/3PGK/3PGK_15.cast', 'r')
	# 	col = 1

	# try:
	# 	path = sys.argv[3]
	# except:
	# 	path='temp'

	cast = [l.rstrip().split() for l in open(castFile, 'r').readlines()]
	# print(cast[0][col])
	# print(col)


	train = []
	test = []

	for i, l in enumerate(cast[1:]):
	# print(seqFile[i*2].rstrip())
		if l[col] == '1':
			train.append([1, i])
		if l[col] == '2':
			train.append([0, i])
		if l[col] == '3':
			test.append([1, i])
		if l[col] == '4':
			test.append([0, i])

	np.savetxt(path+'trainlabels', np.array(train), fmt='%u')
	np.savetxt(path+'testlabels', np.array(test), fmt='%u')


# # Use -run flag as first argument to run as script
# try:
# 	if sys.argv[1] == '-run':
# 		readData(sys.argv[2], sys.argv[3])
# except:
# 	next

if __name__ == "__main__":

	castFile = sys.argv[1]
	col = int(sys.argv[2])
	readData(castFile, col)