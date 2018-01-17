#########################
## Syntax: python3  vector.py  seq_file_path  num_clusters max_iter  outfile  centroid_loadfile(optional)
#########################

import time, sys, math, multiprocessing
import numpy as np
from functools import partial
import stringKmeans as cluster
import fragData

np.random.seed(0)
start_time = time.time()
# ncpus = multiprocessing.cpu_count()
ncpus = 12

def main():

	try: inFile = sys.argv[1]
	except:	inFile = 'sample_data/3PGK_PROTEIN.fasta'

	try: numClusters = int(sys.argv[2])
	except: numClusters = 100

	try: max_iter = float(sys.argv[3])
	except: max_iter = 2

	try: outFile = sys.argv[4]
	except:	outFile = 'temp/kdata'

	try: centroidFile = sys.argv[5]
	except: centroidFile = False

	# Create fragments of len 14 from sequence file
	fragArray = fragData.getData(inFile, 14)[0]
	labels = fragArray[:,0]
	frags = fragArray[:,1]

	print(numClusters, "clusters")

	X = cluster.kmeans(n_clusters=numClusters, load_centroids=centroidFile, max_iter=max_iter, verbose=True)
	X.fit(frags)
	print("Done clustering in %.2f seconds" % (time.time() - start_time))

	np.savetxt('temp/centroids', np.array([c['centroid'] for c in X.clusters]), fmt='%s')

	n_seqs = 1 + int(labels[-1])
	seqFrags = [[] for _ in range(n_seqs)]

	for i, l in enumerate(labels):
		seqNum = int(l)
		seqFrags[seqNum].append(X.allPoints[i])

	# np.save('points.npy', seqFrags)

	pool = multiprocessing.Pool(ncpus)
	func = partial(makeVector, numClusters=X.numClusters)
	kdata = pool.map(func, seqFrags)
	pool.close()
	pool.join()

	np.savetxt(outFile, np.array(kdata), fmt='%.4f')

	print("--- kmeans done %.2f seconds ---" % (time.time() - start_time))


def makeVector(seqFrags, numClusters):

	fragVectors = []
	for frag in seqFrags:
		fragvec = []

		dist = frag['dist']
		fragMean = frag['totaldist'] / numClusters
		for i in range(numClusters):
			if i == frag['cluster']:
				fragvec.append(fragMean + dist)
			else:
				fragvec.append(0)

		fragVectors.append(fragvec)

	# sum pool all frags in seq into one vector
	fv = np.sum(np.array(fragVectors), axis=0)
	return fv


main()
