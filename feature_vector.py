import numpy as np 
import multiprocessing
import sys
from sklearn.cluster import KMeans
from functools import partial

def blast_like_alignment_fast(seq1, seq2):

    score = 0
    seq2htlist = {}

    for i in range(len(seq2)):
        kmer2 = seq2[i:i+3]
        seq2htlist[kmer2] = 1

    for i in range(len(seq1)):
        kmer1 = seq1[i:i+3]

        # if(seq2htlist.get(kmer1) != None):
        if kmer1 in seq2htlist:
            score = score + 1
    return score


def smith_waterman(seq1, seq2):

    m = 5
    mm = -4
    g = -20

    rows = len(seq1) + 1
    cols = len(seq2) + 1

    #initialize V and T
    V = [[0 for i in range(cols)] for j in range(rows)]
    T = [[0 for i in range(cols)] for j in range(rows)]

    maxScore = 0
    maxI = 0
    maxJ = 0

    #do NW recursion
    for i in range(1, rows):
        for j in range(1, cols):

            key = seq1[i-1] + seq2[j-1]
            p = blosum[key]

            d = max(V[i-1][j-1] + p, 0)
            u = max(V[i-1][j] + g, 0)
            l = max(V[i][j-1] + g, 0)

            best = max(d, u, l)

            if d == best:
                V[i][j] = d
                T[i][j] = 'D'
            elif u == best:
                V[i][j] = u
                T[i][j] = 'U'
            elif l == best:
                V[i][j] = l
                T[i][j] = 'L'

            if best > maxScore:
                maxScore = best
                maxI = i
                maxJ = j

    aln_seq1 = ''
    aln_seq2 = ''
    i = maxI
    j = maxJ

    while(V[i][j] > 0):
        if(T[i][j] == 'L'):
            aln_seq1 = '-' + aln_seq1
            aln_seq2 = seq2[j-1] + aln_seq2
            j = j-1
        elif(T[i][j] == 'U'):
            aln_seq1 = seq1[i-1] + aln_seq1
            aln_seq2 = '-' + aln_seq2
            i = i-1
        else:
            aln_seq1 = seq1[i-1] + aln_seq1
            aln_seq2 = seq2[j-1] + aln_seq2
            i = i-1
            j = j-1

    return maxScore


def empirical_kernel(protein):

    fv = []
    with open('reference2', 'r') as ref:

        # seqs = [l.rstrip() for l in ref.readlines() if l[0] != '>']
        # pool = multiprocessing.Pool()
        # # print(seqs[0])
        # partial_blast = partial(blast_like_alignment_fast, protein)
        # fv = pool.map(partial_blast, seqs)

        # l = ref.readline()
        for l in ref.readlines():
            if l[0] != '>':
                l = l.strip('\n')

                score = blast_like_alignment_fast(l, protein.upper())
                # score = smith_waterman(l.upper(), protein.upper())
                # fv = np.append(fv, score)
                fv.append(score)
                # fv.append(str(score))

    return fv
    # return [1.0 , 1.0]


def spectrum_protein(protein):

    fv = []
    k = 3
    AA = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z' ,'X' ]

    for i in range(23**(k)):

        # ind = []
        # kmer = ''

        # for j in range(k):
        #     ind[j] = (i // 23**j) % 23

        # for l in ind:
        #     kmer += AA[l]

        # c = protein.count(kmer)
        # fv.append(str(c))

        firstind = (i // 23**0) % 23
        secondind = (i // 23**1) % 23
        thirdind = (i // 23**2) % 23
        kmer = AA[thirdind]+AA[secondind]+AA[firstind]

        c = protein.count(kmer)
        fv.append(c)

    return fv


def spectrum_dna(dna, k):

    fv = []
    base = ['A','C','G','T']

    for i in range(4**(k)):

        ind = []
        kmer = ''
        
        for j in range(k):
            ind[j] = (i // 4**j) % 4

        for l in ind:
            kmer += base[l]

        c = dna.count(kmer)
        fv.append(str(c))

    return fv


def fragment(sequence, size, stride):

    frags = []

    for i in range(0, len(sequence) - size + 1, stride):
        f = sequence[i:i+size]
        frags.append(f)

    return frags

def kmeans_cluster(data):

    data = np.array(data)
    print(data)
    print(len(data))
    print(len(data[0]))

    kmeans = KMeans(n_clusters=16, random_state=0).fit(data)

    fv = []

    # for c in kmeans:

    #     D = []
    #     mi = np.mean(c)

    #     for j in c:
    #         if j > mi:
    #             D.append(j)

    #     fv.append(str(np.mean(D)))


    # print(kmeans)
    # print(len(kmeans))
    # print(len(kmeans[0]))
    print(kmeans.predict(data))
    return(fv)

# f = open("BLOSUM_62.txt").readlines()
# aa = f[0].split()
# mat = [l.split()[1:] for l in f[1:]]
# mat = [list(map(int, l)) for l in mat]

# blosum  = {}
# for i, m in enumerate(aa):
#     for j, n in enumerate(aa):
#         blosum[m+n] = mat[i][j]


# def SNP(snp):

#     base = ['A','C','G','T']
#     fv = []
#     t = {}
#     n = 0

#     s = snp.split()

#     for i in base:
#         for j in base:
#             k = i+j
#             t[k] = n
#             n += 1
#             # fv.append(str(s.count(k)))

#     for i in s:
#         fv.append(t[i])

#     # print(fv)
#     return fv

# l = fragment('ACGTAGGGATAAGCTA', 5, 1)
# print(l)