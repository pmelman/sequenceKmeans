import os
import sys
import fileinput
import shutil

castFiles = open(sys.argv[1]).readlines()
castFiles = list(map(str.strip, castFiles))
print(castFiles)
# sgeBase = open("sgescript_base")


for cast in castFiles:
	# i = 1

	castPath = '../datasets/CATH/'+cast
	castID = cast[7:13]
	# path = "./submit_files/"
	path = "./"

	cm = [l.rstrip().split() for l in open(castPath, 'r').readlines()]
	ntasks = len(cm[0]) - 1
	tempDir = 'temp_'+castID+'/'
	print(castID)

	sgeFilename = path+"sgescript."+castID
	outfile = "SVMout_"+castID

	open(sgeFilename,'w').close()
	shutil.copy2("sgescript_base", sgeFilename)
	f = open(sgeFilename, 'a')

	f.write('mkdir '+tempDir+'\n')
	f.write('python3 runCast.py '+castPath+' '+str(ntasks)+' temp/kdata '+tempDir+' >> '+outfile+'\n')




	# f.write('"cp" -r ~/thesis/stringKmeans/ ~/thesis/strKtemp_'+d+'\n')
	# f.write('cd ~/thesis/strKtemp_'+d+'\n')
	# f.write('echo '+d+' | tee '+outfile+'\n')
	# f.write('python3 fragData.py '+d+' | tee '+outfile+'\n')
	# f.write('python3 stringKmeans.py 200 | tee '+outfile+'\n')
	# f.write('python3 SVM.py | tee '+outfile+'\n')

	# f.write('cp '+outfile+' ~/thesis/results/ \n')
	# f.write('rm -r ~/thesis/strKtemp_'+d)
	f.write('rm -r '+tempDir)
	f.close()

	os.system("qsub "+sgeFilename)
	# os.system("rm "+sgeFilename)
