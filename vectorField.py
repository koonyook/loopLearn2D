import numpy as np
from numpy import ma
import random

import matplotlib
import os
if os.name != 'nt':
	matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def plotDeformedGrid2D(reconstructor, gridSpace=0.1, dataPoints=None, filename=None, fig=None):
	
	lineResolution = 200
	horizontalLines=[]
	
	for vx in np.arange(-1.0, 1.01, gridSpace):
		for vy in np.linspace(-1.0,1.0, num=lineResolution):
			horizontalLines.append([vx,vy])

	hLines=np.array(horizontalLines)
	vLines=np.vstack([hLines[:,1],hLines[:,0]]).transpose()	#just swap x and y

	#print(hLines.shape)
	#print(vLines.shape)

	allLines=np.vstack([hLines,vLines])

	X1, Y1 = reconstructor(allLines[:,0],allLines[:,1])

	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch
	
	if(dataPoints!=None):
		plt.scatter(dataPoints[:,0],dataPoints[:,1],color='r',s=1)
		RX,RY=reconstructor(dataPoints[:,0],dataPoints[:,1])
		plt.scatter(RX,RY,color='g',s=1)

	plt.axis('equal')	#allow true square scaling
	#plt.quiver(X0, Y0, U, V, scale=1, units='x')

	color='grey'
	for i in range(len(X1)/lineResolution):
		if(i==len(X1)/lineResolution/2):
			color='brown'
		plt.plot(X1[i*lineResolution:(i+1)*lineResolution],Y1[i*lineResolution:(i+1)*lineResolution], c=color)

	plt.axis([-1.1,1.1,-1.1,1.1])

	if(filename==None):
		plt.show()
		#print("what")
	else:
		plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
		fig.clf()

def plotVectorField2D(reconstructor, dataPoints=None, filename=None, fig=None, capVectorLength=False):
	resolution=0.05
	X0, Y0 = np.meshgrid(np.arange(-1, 1.01, resolution), np.arange(-1, 1.01, resolution))
	X0=X0.flatten()
	Y0=Y0.flatten()
	X1, Y1 = reconstructor(X0,Y0)
	U=X1-X0
	V=Y1-Y0

	M = np.hypot(U, V)	#put color in first

	#then, scale U, V to avoid the dot
	for i in range(len(M)):
		if(M[i]<0.02):
			U[i]=U[i]/M[i]*0.02
			V[i]=V[i]/M[i]*0.02
		elif(M[i]>0.05 and capVectorLength):
			U[i]=U[i]/M[i]*0.05
			V[i]=V[i]/M[i]*0.05

	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch
	
	
	if(dataPoints!=None):
		plt.scatter(dataPoints[:,0],dataPoints[:,1],color='r',s=1)
		RX,RY=reconstructor(dataPoints[:,0],dataPoints[:,1])
		plt.scatter(RX,RY,color='g',s=1)

	plt.axis('equal')	#allow true square scaling
	plt.quiver(X0, Y0, U, V, M, scale=1, units='x')

	plt.axis([-1.1,1.1,-1.1,1.1])

	if(filename==None):
		plt.show()
		#print("what")
	else:
		plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
		fig.clf()

def plot2VectorField(reconstructor, dataPoints=None, filename=None, fig=None):
	resolution=0.05
	X0, Y0 = np.meshgrid(np.arange(-1, 1.01, resolution), np.arange(-1, 1.01, resolution))
	X0=X0.flatten()
	Y0=Y0.flatten()
	X1_recon,Y1_recon,X1_predict,Y1_predict = reconstructor(X0,Y0)
	
	U_recon=X1_recon-X0
	V_recon=Y1_recon-Y0

	U_predict=X1_predict-X0
	V_predict=Y1_predict-Y0

	if(fig==None):
		fig=plt.figure(figsize=(14, 6))	#in inch
	
	ax1 = plt.subplot2grid((1,2), (0,0))

	if(dataPoints!=None):
		plt.scatter(dataPoints[:,0],dataPoints[:,1],color='r',s=1)

	plt.axis('equal')	#allow true square scaling
	plt.quiver(X0, Y0, U_recon, V_recon, scale=1, units='x')

	plt.axis([-1.1,1.1,-1.1,1.1])

	####################################

	ax2 = plt.subplot2grid((1,2), (0,1))

	if(dataPoints!=None):
		plt.scatter(dataPoints[:,0],dataPoints[:,1],color='r',s=1)

	plt.axis('equal')	#allow true square scaling
	plt.quiver(X0, Y0, U_predict, V_predict, scale=1, units='x')

	plt.axis([-1.1,1.1,-1.1,1.1])

	if(filename==None):
		plt.show()
		#print("what")
	else:
		plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
		fig.clf()

def noRecon(X0,Y0):
	return X0,Y0

def simpleRecon(X0,Y0):
	X1=[]
	Y1=[]
	for x in X0:
		X1.append(x+random.uniform(-0.1, 0.1))
	for y in Y0:
		Y1.append(y+random.uniform(-0.1, 0.1))

	return X1,Y1

def advanceRecon(X0,Y0):
	X1=[]
	Y1=[]
	X2=[]
	Y2=[]
	for x in X0:
		X1.append(x+random.uniform(-0.1, 0.1))
		X2.append(x+random.uniform(-0.1, 0.1))
	for y in Y0:
		Y1.append(y+random.uniform(-0.1, 0.1))
		Y2.append(y+random.uniform(-0.1, 0.1))
	return X1,Y1,X2,Y2

if __name__ == '__main__':
	from dataGenerator import sampleFromCircle
	plotDeformedGrid2D(noRecon)
	#plotVectorField2D(simpleRecon,sampleFromCircle(0.75,100))
	#plot2VectorField(advanceRecon,sampleFromCircle(0.75,100),'test.png')
	#print("done")
	#plotVectorField2D(simpleRecon,'testUbuntu.png')