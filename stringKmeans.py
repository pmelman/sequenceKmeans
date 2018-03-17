import multiprocessing
import numpy as np
from functools import partial


ncpus = multiprocessing.cpu_count()
# ncpus = 2

class kmeans(object):

	def __init__(self, n_clusters=40, load_centroids=False, cutoff=1, max_iter=50, verbose=False):

		self.clusters = []
		self.cutoff = cutoff
		self.numClusters = n_clusters
		self.maxLoops = max_iter
		self.verbose = verbose

		if load_centroids == False:
			self.centroids = None
		else:
			try:
				self.centroids = [line.rstrip() for line in open(load_centroids).readlines()]
				k = len(self.centroids)
				if k != self.numClusters:
					print("Found %s centroids, setting n_clusters to %s" % (k, k))
					self.numClusters = k
			except:
				print("Could not load centroids, using random centers")
				self.centroids = None


	def updateClusters(self, tempClusters):

		oldCentroids = [cluster['centroid'] for cluster in self.clusters]
		tempClusters = [(tClust, self.centroids[i]) for i, tClust in enumerate(tempClusters)]

		pool = multiprocessing.Pool(ncpus)
		newCentroids = pool.map(calculateCentroid, tempClusters)
		pool.close()
		pool.join()

		self.centroids = newCentroids
		for idx, c in enumerate(tempClusters):
			self.clusters[idx]['centroid'] = newCentroids[idx]
			self.clusters[idx]['points'] = c

		biggestShift = 0
		for i, cent in enumerate(newCentroids):
			shift = hammingDistance(oldCentroids[i], cent)
			if shift > biggestShift:
				biggestShift = shift

		return biggestShift


	def initialize(self):

		if self.centroids != None:
			for i in range(self.numClusters):
				cluster = {}
				cluster['points'] = []
				cluster['centroid'] = self.centroids[i]
				self.clusters.append(cluster)
		else:
			initial = np.random.choice(self.allPoints, size=self.numClusters, replace=False)
			self.centroids = []
			for p in initial:
				cluster = {}
				cluster['points'] = [p]
				cluster['centroid'] = p['val']
				self.centroids.append(cluster['centroid'])
				self.clusters.append(cluster)


	def fit(self, data):

		points = [{'val': data[i], 'cluster': -1} for i in range(len(data))]
		self.allPoints = points
		self.initialize()

		# Loop through the dataset until the clusters stabilize or maxLoops reached
		loopCounter = 0
		while True:
			loopCounter += 1
			tempClusters = [[] for _ in self.clusters]

			pool = multiprocessing.Pool(ncpus)

			func = partial(updatePoint, centroids=self.centroids)
			tempPoints = pool.map(func, self.allPoints, chunksize=20)
			self.allPoints = tempPoints

			pool.close()
			pool.join()

			for p in self.allPoints:
				tempClusters[p['cluster']].append(p)

			biggestShift = self.updateClusters(tempClusters)

			meanDist = sum([p['dist'] for p in self.allPoints]) / len(self.allPoints)

			if self.verbose: print('mean dist: %.5f  max shift: %s' % (meanDist, biggestShift))

			if biggestShift < self.cutoff:
				print("Converged after %s iterations" % loopCounter)
				self.predict = [p['cluster'] for p in self.allPoints]
				break

			if loopCounter >= self.maxLoops:
				self.predict = [p['cluster'] for p in self.allPoints]
				break


	def transform(self):
	# For each point, return array of distances to each centroid

		pool = multiprocessing.Pool(ncpus)

		func = partial(allDistances, centroids=self.centroids)
		Tdata = pool.map(func, self.allPoints, chunksize=20)

		pool.close()
		pool.join()

		return Tdata


def allDistances(p, centroids):

	distances = []

	for i, centroid in enumerate(centroids):
		distance = blosumDistance(p['val'], centroid)
		# distance = hammingDistance(p['val'], centroid)
		distances.append(distance)

	return np.array(distances, dtype=np.float16)


def updatePoint(p, centroids):

	clusterIndex = -1
	smallestDistance = 10000000
	top3 = [[100000, -1], [100000, -1], [100000, -1]]

	totalDist = 0
	for i, centroid in enumerate(centroids):
		distance = blosumDistance(p['val'], centroid)
		# distance = hammingDistance(p['val'], centroid)
		totalDist+= distance

		if distance < smallestDistance:
			smallestDistance = distance
			clusterIndex = i

		if distance < top3[-1][0]:
			del top3[-1]
			top3.append([distance, i])
			top3.sort()

	p['dist'] = smallestDistance
	p['totaldist'] = totalDist
	p['cluster'] = clusterIndex
	p['top3'] = top3
	return p


def calculateCentroid(tempCluster):
# find the centroid by taking the mode of every character position

	points = tempCluster[0]
	oldCentroid = tempCluster[1]

	AA = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X']
	if len(points) > 0:
		mseq = ''
		for i in range(len(points[0]['val'])):

			coloumnChars = [p['val'][i] for p in points]
			chCount = [0 for _ in range(23)]

			for ch in coloumnChars:
				idx = AA.index(ch)
				chCount[idx] += 1

			mostCommon = AA[np.argmax(chCount)]
			mseq += mostCommon
	else:
		mseq = oldCentroid

	return mseq


def hammingDistance(a, b):
# Hamming distance, count mismatches
	h = 0
	for i in range(len(a)):
		if(a[i] != b[i]):
			h += 1
	return h


def blosumDistance(a, b):
# Returns blosum distance, which is negative blosum score

	d = 0
	for i in range(len(a)):
		key = a[i] + b[i]
		p = blosum[key]
		d += p
	return -d /len(a)


f = open("BLOSUM_62.txt").readlines()
aa = f[0].split()
mat = [l.split()[1:] for l in f[1:]]
mat = [list(map(int, l)) for l in mat]

blosum  = {}
for i, m in enumerate(aa):
    for j, n in enumerate(aa):
        blosum[m+n] = mat[i][j]

