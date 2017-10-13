import sys
import numpy as np
import multiprocessing
import time
import csv
import warnings
import feature_vector
import _pickle
from Bio import SeqIO, BiopythonWarning
# warnings.simplefilter('ignore', BiopythonWarning)

start_time = time.time()

def obtain_feature_vector(seq):

	try:
		fragLen = int(sys.argv[2])
	except:
		fragLen = 14

	if len(seq) < fragLen:
		print('padded sequence len ', len(seq))
		seq = seq.ljust(fragLen, 'X')
	frags = feature_vector.fragment(seq, fragLen, 1)

	return frags


try:
	data = sys.argv[1]
except:
	data = 'temp/seq_data'
	# data = '../3PGK/3PGK_PROTEIN.fasta'

allData = []
row = 0
seqNum = []

file = SeqIO.parse(data, 'fasta')
seqs = [l for l in file]
names = [s.id for s in seqs]

for i, s in enumerate(seqs):

	frags = obtain_feature_vector(str(s.seq))
	for f in frags:
		seqNum.append(row)
		allData.append(f.upper())
	row += 1

np.savetxt('temp/data', np.array(allData), fmt='%s')
np.savetxt('temp/names', np.array(names), fmt='%s')
np.savetxt('temp/labels', np.array(seqNum), fmt='%u')

# print("--- frag done %s seconds ---" % (time.time() - start_time))
print("frag done:", len(allData), "frags")
