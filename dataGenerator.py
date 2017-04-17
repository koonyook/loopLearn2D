import random
import numpy as np

def sampleFromCircle(radius,trainsetSize):
	ans=np.zeros([trainsetSize,2])
	for i in range(trainsetSize):
		angle=random.uniform(0,2*np.pi)
		ans[i]=radius*np.cos(angle),radius*np.sin(angle)

	return ans

def sampleFromCircleFlow(radius, rotationRad, trainsetSize):
	ans=np.zeros([trainsetSize,2])
	shiftedAns=np.zeros([trainsetSize,2])
	for i in range(trainsetSize):
		angle=random.uniform(0,2*np.pi)
		ans[i]=radius*np.cos(angle),radius*np.sin(angle)

		angle=angle+rotationRad
		shiftedAns[i]=radius*np.cos(angle),radius*np.sin(angle)

	return ans,shiftedAns

def deformCircleToInf(data):
	ans=np.zeros([len(data),2])
	for i in range(len(data)):
		x,y=data[i]
		ans[i]=x,x*y*2.5

	return ans

def sampleFromInfinityFlow(trainSize):
	beginPoint,endPoint=sampleFromCircleFlow(0.75,10.0*np.pi/180.0,trainSize)
	beginPoint=deformCircleToInf(beginPoint)
	endPoint=deformCircleToInf(endPoint)
	return beginPoint,endPoint


def gen():
	t=np.linspace(0.0, 1.0, num=100)
	t=t*12+3	#change from [0,1] to [3,15]
	return t*0.04*np.sin(t), t*0.04*np.cos(t)

def circleTransform(t):
	return 0.75*np.cos(t*2*np.pi),0.75*np.sin(t*2*np.pi)

def ellipseTransform(t):
	x=np.sin(t*2*np.pi+0.75)
	y=np.cos(t*2*np.pi)
	return 0.75*x,0.75*y

def beanTransform(t):
	x=np.sin(t*2*np.pi+0.75)
	y=np.cos(t*2*np.pi)
	return 0.75*x,0.75*y*y*y

def sTransform(t):
	x=np.sin(t*2*np.pi)
	y=t-0.5
	return 0.75*x,1.5*y

def spiralTransform(t):
	t=t*12+3	#change from [0,1] to [3,15]
	return t*0.04*np.sin(t), t*0.04*np.cos(t)

def sampleFromFunction(transformer,trainSize,stepForward=0.02):	#stepForward in range 0-1
	t0=np.random.uniform(0.0,1.0,trainSize)
	t1=t0+stepForward
	x0,y0=transformer(t0)
	x1,y1=transformer(t1)

	return np.vstack([x0,y0]).transpose(), np.vstack([x1,y1]).transpose()




def sampleFromBean(trainSize,stepForward=0.02):	#stepForward in range 0-1
	t0=np.random.uniform(0.0,1.0,trainSize)
	t1=t0+stepForward
	x0,y0=beanTransform(t0)
	x1,y1=beanTransform(t1)

	return np.vstack([x0,y0]).transpose(), np.vstack([x1,y1]).transpose()

def addGaussianNoise(patches,sd=0.06):
	return patches+np.random.normal(0,sd,size=patches.shape)

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	fig=plt.figure(figsize=(6, 6))
	x,y=gen()
	plt.scatter(x,y)
	plt.axis('equal')
	plt.show()
	
	#a=sampleFromCircle(0.75,100)
	#a=deformCircleToInf(a)
	#plt.scatter(a[:,0],a[:,1])
	#plt.show()

	#time=np.linspace(0.0,1.0,100)
	#x,y=beanTransform(time)

	#plt.plot(time,y)
	#plt.show()
	


	'''
	#a,b=sampleFromCircleFlow(0.75,10.0*np.pi/180.0, 100)
	#a,b=sampleFromInfinityFlow(100)
	#a,b=sampleFromBean(100)
	a,b=sampleFromFunction(ellipsTransform,100)
	#print(len(a))
	plt.axis('equal')	#allow true square scaling
	v=b-a
	plt.quiver(a[:,0], a[:,1], v[:,0], v[:,1], scale=1, units='x')
	plt.axis([-1.1,1.1,-1.1,1.1])
	plt.show()
	'''
	