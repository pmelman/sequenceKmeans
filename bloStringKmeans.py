import random, time, sys, math
# import time
# import sys
# import math
import numpy as np
import pandas as pd
import multiprocessing
from collections import Counter
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(7)
np.random.seed(7)
start_time = time.time()

def main():

	numClusters = int(sys.argv[1])
	try: arg2 = float(sys.argv[2])
	except: arg2 = 50

	print(numClusters, "clusters")

	f = open('temp/data').readlines()
	frags = [i.rstrip() for i in f]

	X = kmeans(n_clusters=numClusters, cutoff=1, load_centroids=False, max_loops=arg2)
	X.fit(frags)

	np.savetxt('temp/centroids', np.array([c['centroid'] for c in X.clusters]), fmt='%s')

	labels = np.genfromtxt('temp/labels')
	n_seqs = 1 + int(labels[-1])
	seqFrags = [[] for _ in range(n_seqs)]
	for i, l in enumerate(labels):
		seqNum = int(l)
		seqFrags[seqNum].append(X.allPoints[i])

	# call makeVector for each sequence index
	pool = multiprocessing.Pool()
	kdata = pool.map(makeVector, seqFrags, chunksize=8)
	pool.close()
	pool.join()
	# kdata = [makeVector(s) for s in seqFrags]

	np.savetxt('temp/kdata', np.array(kdata), fmt='%.4f')

	unique, counts = np.unique(X.predict, return_counts=True)
	plt.plot(unique, counts, 'ro')
	plt.savefig('bar.png')
	# print(dict(zip(unique, counts)))

	print("--- kmeans done %s seconds ---" % (time.time() - start_time))

class kmeans(object):

	def __init__(self, n_clusters=8, load_centroids=False, cutoff=1, dist_function='Hamming', max_loops=50):
		
		self.clusters = []
		self.cutoff = cutoff
		self.numClusters = n_clusters
		self.maxLoops = max_loops

		if load_centroids == False:
			self.centroids = None
		else:
			try:
				self.centroids = [line.rstrip() for line in open(self.centroids).readlines()]
				k = len(self.centroids)
				if k != self.numClusters:
					print("Found %s centroids, setting n_clusters to %s" % (k, k))
					self.numClusters = k
			except:
				print("Could not load centroids, using random centers")
				self.centroids = None

	def updateCluster(self, idx, points):
	# calculate new centroid and return distance between new and old

		oldCentroid = self.clusters[idx]['centroid']
		newCentroid = self.calculateCentroid(self.clusters[idx], points)

		self.clusters[idx]['centroid'] = newCentroid
		self.clusters[idx]['points'] = points
		self.centroids[idx] = newCentroid
		# subtract identity distance for BLOSUM compatability
		shift = abs(getDistance(oldCentroid, self.clusters[idx]['centroid']) - getDistance(oldCentroid, oldCentroid))
		return shift

	def calculateCentroid(self, cluster, points):
	# find the centroid by taking the mode of every character position

		if len(points) > 0:
			mseq = ''
			for i in range(len(points[0]['val'])):

				coloumnChars = [p['val'][i] for p in points]

				chCount = [0 for _ in range(23)]
				AA = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X']

				for ch in coloumnChars:
					idx = AA.index(ch)
					chCount[idx] += 1

				mostCommon = AA[np.argmax(chCount)]
				# mostCommon = Counter(coloumnChars).most_common()[0][0]
				mseq += mostCommon
		else:
			mseq = cluster['centroid']
		return mseq

	def initialize(self):

		if self.centroids != None:
			for i in range(self.numClusters):
				cluster = {}
				cluster['points'] = []
				cluster['centroid'] = self.centroids[i]
				self.clusters.append(cluster)
		else:
			initial = random.sample(self.allPoints, self.numClusters)
			self.centroids = []
			for p in initial:
				cluster = {}
				cluster['points'] = [p]
				cluster['centroid'] = p['val']
				self.centroids.append(cluster['centroid'])
				self.clusters.append(cluster)


	def fit(self, data):

		points = [{'val': data[i], 'distance': np.array([-1 for _ in range(self.numClusters)], dtype=np.int16), 'cluster': -1} for i in range(len(data))]
		self.allPoints = points
		self.initialize()

		# Loop through the dataset until the clusters stabilize
		loopCounter = 0
		while True:
			# Create a list of lists to hold the points in each cluster
			loopCounter += 1
			tempClusters = [[] for _ in self.clusters]

			pool = multiprocessing.Pool()

			func = partial(updatePoint, centroids=self.centroids)
			tempPoints = pool.map(func, self.allPoints, chunksize=20)
			self.allPoints = tempPoints

			pool.close()
			pool.join()

			for p in self.allPoints:
				tempClusters[p['cluster']].append(p)

			biggestShift = 0
			for i, c in enumerate(self.clusters):
				shift = self.updateCluster(i, tempClusters[i])
				biggestShift = max(biggestShift, shift)

			print('mean dist: %.4f  shift: %s' % (np.mean([p['distance'][p['cluster']] for p in self.allPoints]), biggestShift))

			if biggestShift < self.cutoff:
				print("Converged after %s iterations in %s seconds" % (loopCounter, time.time() - start_time))
				self.predict = [p['cluster'] for p in self.allPoints]
				break

			if loopCounter >= self.maxLoops:
				self.predict = [p['cluster'] for p in self.allPoints]
				break

def updatePoint(p, centroids):
	clusterIndex = -1
	smallestDistance = 10000000

	for i, c in enumerate(centroids):
		distance = getDistance(p['val'], c)
		p['distance'][i] = distance
		if distance < smallestDistance:
			smallestDistance = distance
			clusterIndex = i

	p['cluster'] = clusterIndex
	return p

def makeVector(seq):

	frags = []
	for f in seq:
		fragvec = []
		Tdata = f['distance']
		fragMean = np.mean(Tdata)
		for i, n in enumerate(Tdata):
			# fragvec.append(max(0, fragMean + n))

			if n == min(Tdata):
				fragvec.append(fragMean + n)
				# fragvec.append(1)
			else:
				fragvec.append(0)
		frags.append(fragvec)

	# sum pool all frags in seq into one vector
	fv = np.sum(np.array(frags), axis=0)
	return fv


# Hamming distance, count mismatches
def getDistance(a, b):
# 	h = 0
# 	for i in range(len(a)):
# 		if(a[i] != b[i]):
# 			h += 1
# 	return h

# def blosumDist(a, b):	
	d = 0
	for i in range(len(a)):
		key = a[i] + b[i]
		# key = "".join([a[i], b[i]])
		p = blosum[key]
		d += p
	return -d
	# return 1 / math.exp(d / len(a))



f = open("BLOSUM_62.txt").readlines()
aa = f[0].split()
mat = [l.split()[1:] for l in f[1:]]
mat = [list(map(int, l)) for l in mat]

blosum  = {}
for i, m in enumerate(aa):
    for j, n in enumerate(aa):
        blosum[m+n] = mat[i][j]


main()
