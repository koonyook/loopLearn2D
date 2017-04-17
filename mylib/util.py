import numpy as np

def randomNewTrainingBatch(scopeFile,kernelSize,batchSize):
	data=np.load(scopeFile)

	nSet=data.shape[0]
	nIn=[]
	for i in range(nSet):
		nIn.append(len(data[i]))

	#randomly pick up patch
	patches=[]
	for i in range(batchSize):
		r0=np.random.randint(nSet)
		r1=np.random.randint(nIn[r0]-kernelSize)
		onePatch=data[r0][r1:r1+kernelSize]
		patches.append(np.array(onePatch).flatten())

	patches=np.array(patches)

	return patches


def extractAllConsecutivePatches(testFile,kernelSize):
	#generate all possible patches in sequence
	testdata=np.load(testFile)
	nSet=testdata.shape[0]
	nIn=[]

	for i in range(nSet):
		nIn.append(len(testdata[i]))

	#print nIn

	testPatches=[]
	for i in range(nSet):
		for j in range(len(testdata[i])-kernelSize):
			onePatch=testdata[i][j:j+kernelSize]
			testPatches.append(np.array(onePatch).flatten())

	return np.array(testPatches)