import numpy as np
import math

import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

def showROC(n,p,label='',color='b',show=True):
	#augment negative with -1, and positive with +1
	sizep=p.size
	sizen=n.size

	p=np.vstack((p,np.ones(sizep)))
	n=np.vstack((n,-np.ones(sizen)))

	a=np.hstack((p,n))
	
	#sort in assending order
	a=a[:,a[0,:].argsort()]
	a=np.fliplr(a)	#flip to be descending order (when positive group are on the higher side)

	fpr=[0]
	tpr=[0]

	tpCount=0
	fpCount=0

	for i in range(sizen+sizep):
		if(a[1,i]==1):	#positive found
			tpCount=tpCount+1
		else:	#negative found
			fpCount=fpCount+1

		fpr.append(float(fpCount)/sizen)
		tpr.append(float(tpCount)/sizep)

	#add final piece
	fpr.append(1.0)
	tpr.append(1.0)

	print 'AUC=',
	print np.trapz(tpr,fpr)

	plt.plot(fpr,tpr,label=label,color=color)
	if(show):
		plt.legend(loc='lower right')
		plt.xlabel('False positive rate')
		plt.ylabel('True positive rate')
		plt.show()


if __name__ == "__main__":
	negative=np.random.rand(100) #negative group (normal group)
	positive=np.random.rand(100)+0.6 #positive group (anomaly group)

	showROC(negative,positive,label='hey')