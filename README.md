# sequenceKmeans
This is a tool for performing K-means clustering on a set of protein strings. The scripts contained here can be used to transform a set of protein sequences into a set of feature vectors by using K-means feature learning. 

stringKmeans.py is a stand-alone module whose syntax is based upon the scikit-learn kmeans function. 

I have provided some sample data to demonstrate how it works. To run with default parameters on the sample data, type these commands:
python3 vectorScriptExample.py
python3 runCast.py

# References
The cast matrix and data files were obtained from http://pongor.itk.ppke.hu/benchmark/#/Browse

The original source of the protein data:

Pollack, J.D., Li, Q. and Pearl, D.K. (2005) Taxonomic utility of a phylogenetic analysis of phosphoglycerate kinase proteins of Archaea, Bacteria, and Eukaryota: insights by Bayesian analyses, Mol Phylogenet Evol, 35, 420-430. 
