import sys
import numpy as np
from Bio import SeqIO

def getData(inFile, fragLen, stride=1):

	seqs = [line for line in SeqIO.parse(inFile, 'fasta')]
	names = [(s.id, len(s)) for s in seqs]

	allData = []
	seqIdx = 0

	for s in seqs:
		seq = str(s.seq).upper()
		frags = fragment(seq, fragLen, stride)
		for f in frags:
			allData.append([seqIdx, f.upper()])
		seqIdx += 1

	
	return np.array(allData), np.array(names)

def fragment(seq, fragLen, stride):

	# If seq is too short, pad with X (unknown amino acid)
	if len(seq) < fragLen:
		print('padded sequence len ', len(seq))
		seq = seq.ljust(fragLen, 'X')

	frags = []

	for i in range(0, len(seq) - fragLen + 1, stride):
		f = seq[i:i+fragLen]
		frags.append(f)

	return frags

if __name__ == "__main__":

	try: inFile = sys.argv[1]
	except: inFile = 'sample_data/3PGK_PROTEIN.fasta'

	try: fragLen = int(sys.argv[2])
	except: fragLen = 14

	data, names = getData(data, fragLen)

	np.savetxt('temp/data', data, fmt='%s')
	np.savetxt('temp/names', names, fmt='%s')

