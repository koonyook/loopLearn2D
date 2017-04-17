import numpy as np
import math

def loopWindowAverageFilter(data,windowSize):
	#windowSize should be an odd number
	n=len(data)
	output=np.zeros(n)
	half=(windowSize-1)/2
	for i in range(n):
		s=0.0
		for j in xrange(-half,half+1):
			s=s+data[(i+j)%n]
		output[i]=s/windowSize

	return output

def circularMeanSD(data):
	#data is a numpy array of floats in range [0,1)
	
	#first, estimate mean using Cartesian coordinate
	angle=data*(2*math.pi)
	y=np.sin(angle)
	x=np.cos(angle)
	estimatedMean=math.atan2(sum(y),sum(x))/(2*math.pi) #this is just a rough estimation
	estimatedMean=np.fmod(estimatedMean+1.0,1.0)	#keep it positive
	#print estimatedMean

	#rotate the value forward
	fwdShift=1.5-estimatedMean
	shiftedData=np.fmod(data+fwdShift,1.0)	#shifted data should be dense around 0.5

	mean=np.fmod(np.mean(shiftedData)+(2.0-fwdShift),1.0)	#fmod will return negative value if input is negative
	std=np.std(shiftedData)

	return mean,std

if __name__ == "__main__":
	#print loopWindowAverageFilter([1,2,3,4,5],3)
	d=np.fmod(np.random.rand(100)*0.5 +0.25,1.0)
	print d
	print circularMeanSD(d)