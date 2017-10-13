import sys
import numpy as np
import multiprocessing
import time
import csv
import feature_vector
import warnings
from Bio import SeqIO
from Bio import BiopythonWarning
# warnings.simplefilter('ignore', BiopythonWarning)

start_time = time.time()

try:
	data = sys.argv[1]
except:
	data = 'temp/seq_data'
	# data = '../3PGK/3PGK_PROTEIN.fasta'

allData = []

file = SeqIO.parse(data, 'fasta')
seqs = [l for l in file]
names = [s.id for s in seqs]
allData = [str(s.seq) for s in seqs]

pool = multiprocessing.Pool(10)

# if sys.argv[1] == 'spectrum':
# 	print('spectrum')
# 	kernelized = pool.map(feature_vector.spectrum_protein, allData)
# 	pool.close()
# 	pool.join()
# elif sys.argv[1] == 'empirical':
# 	print('empirical')

kernelized = pool.map(feature_vector.empirical_kernel, allData)
pool.close()
pool.join()
# kernelized = [feature_vector.empirical_kernel(i) for i in allData]

np.savetxt('temp/data.csv', np.array(kernelized), fmt='%u')


print("--- %s seconds ---" % (time.time() - start_time))
