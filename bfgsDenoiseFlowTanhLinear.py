#this code try to replicate the result of spiral reconstruction written in a paper
#2014 What Regularized Auto-Encoders Learn from the Data-Generating Distribution [alain,bengio]

import sys
import os
#sys.path.append('/home/koonyook/Dropbox/research/code/python')

import tensorflow as tf
import numpy as np
import math

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
#import matplotlib.gridspec as gridspec

from scipy.optimize import minimize		# to use BFGS optimizer

from mylib import roc
from mylib import mymath as mm

from vectorField import *
from dataGenerator import *

import random

#============parameters=============
folder='circleDenoiseFlowLBFGS_25h_0.04g/'	#this folder will be generated automatically
selectedTransform=circleTransform 	#choose between circleTransform,ellipseTransform,beanTransform

#===================================

trainsetFutureFile=folder+"trainsetFuture.dat"
trainsetNoisyFile=folder+"trainsetNoisy.dat"
trainsetSize=10000

#genNewData=False

#stocasticTraining=False
#roundPerTrainingSet=4000

sessionFile=folder+"tanhGD.ckpt"


logDirectory=folder+'/logGD/'
imageFolder=folder+'/img/'

if not os.path.exists(folder):
    os.makedirs(folder)
    genNewData=True
    startNewTraining=True
else:
	genNewData=False
	startNewTraining=False


if not os.path.exists(logDirectory):
    os.makedirs(logDirectory)

if not os.path.exists(imageFolder):
    os.makedirs(imageFolder)

runTraining=True

hiddenSize=25
patchSize=2

#=========================

if genNewData and runTraining:
	patches,futurePatches=sampleFromFunction(selectedTransform,trainsetSize)
	noisyPatches = addGaussianNoise(patches,0.04)
	patches.dump(trainsetFutureFile)
	noisyPatches.dump(trainsetNoisyFile)
else:
	futurePatches=np.load(trainsetFutureFile).astype(np.float32)
	noisyPatches=np.load(trainsetNoisyFile).astype(np.float32)
#=========================

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, patchSize]) #input=[batch size,flatten patch]
expectedOutput = tf.placeholder(tf.float32, shape=[None, patchSize] )	#same size as x

weightInitRange=math.sqrt(6.0/(patchSize+patchSize+1))

global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

#name is important if you want to save the session
W1 = tf.Variable(tf.random_uniform([patchSize,hiddenSize], -weightInitRange, weightInitRange),name="W1")
b1 = tf.Variable(tf.zeros([hiddenSize]),name="b1")
W2 = tf.Variable(tf.random_uniform([hiddenSize,patchSize], -weightInitRange, weightInitRange),name="W2")
b2 = tf.Variable(tf.zeros([patchSize]),name="b2")

hidden=tf.tanh(tf.matmul(x,W1)+b1,name='hidden')

y = tf.matmul(hidden,W2)+b2

#LAMBDA=0.0001
#BETA=3.0
#roh = 0.01 #0.01 #sparsityParam	#for 25 hidden nodes, I think 0.04 will allow a hidden node to be fully activated (0.01 might be good for 100 hidden nodes)
#contraction_level=0.05

sample_error = tf.reduce_sum(tf.square(y-expectedOutput),reduction_indices=1)	#reconstruction error of each individual sample
square_error =  tf.mul(0.5,tf.reduce_mean(sample_error), name='square_error')
#weight_decay = tf.mul(LAMBDA*0.5,(tf.reduce_sum(tf.square(W1))+tf.reduce_sum(tf.square(W2))), name='weight_decay')
#roh_hat = tf.reduce_mean(hidden,reduction_indices=0, name='roh_hat')
#sparsity_penalty = tf.mul(BETA,tf.reduce_sum(roh*tf.log(roh/roh_hat)+(1.0-roh)*tf.log((1.0-roh)/(1.0-roh_hat))), name='sparsity_penalty')

#jacobian = tf.reshape(hidden*(1.0-hidden),[-1,hiddenSize,1])*tf.reshape(tf.transpose(W1),[1,hiddenSize,patchSize])
#contractive_penalty = tf.reduce_mean(tf.square(jacobian), name='contractive_penalty')

#cost = tf.add_n([square_error ,weight_decay, sparsity_penalty], name='cost')
#cost = tf.add_n([square_error, contraction_level*contractive_penalty], name='cost')
cost = tf.add_n([square_error], name='cost')

W1_grad,b1_grad,W2_grad,b2_grad = tf.gradients(cost, [W1,b1,W2,b2])

#optimizer = tf.train.GradientDescentOptimizer(0.3)
#optimizer = tf.train.AdamOptimizer(0.1)
#optimizer = tf.train.AdadeltaOptimizer(0.1)
##optimizer = tf.train.AdagradOptimizer(1.5)
#optimizer = tf.train.MomentumOptimizer(0.4,0.6)
#optimizer = tf.train.FtrlOptimizer(1.5)
##optimizer=tf.train.RMSPropOptimizer(0.01)

#train = optimizer.minimize(cost,global_step=global_step_tensor)

#rearrange weight to see what it is learning
#W_Grid = tf.reshape(tf.transpose(W1),[hiddenSize,kernelSize,inChannelSize])

saver = tf.train.Saver()

def autoencoderReconstruction(X0,Y0):
	gridInput=np.vstack((X0,Y0)).transpose()
	gridOutput=sess.run(y,feed_dict={x:gridInput})
	return gridOutput[:,0],gridOutput[:,1]

def serialize(W1_,b1_,W2_,b2_):
	return np.hstack([W1_.flatten(),b1_.flatten(),W2_.flatten(),b2_.flatten()])

def deserialize(params):
	ind=0
	W1_=params[ind:ind+patchSize*hiddenSize].reshape((patchSize,hiddenSize))
	ind=ind+patchSize*hiddenSize
	b1_=params[ind:ind+hiddenSize]
	ind=ind+hiddenSize

	W2_=params[ind:ind+hiddenSize*patchSize].reshape((hiddenSize,patchSize))
	ind=ind+hiddenSize*patchSize
	b2_=params[ind:ind+patchSize]

	return W1_,b1_,W2_,b2_

'''
def objective(params):
	W1_,b1_,W2_,b2_=deserialize(params)
	return sess.run(cost, feed_dict={W1:W1_,b1:b1_,W2:W2_,b2:b2_,x:noisyPatches,expectedOutput:patches})

def gradient(params):
	W1_,b1_,W2_,b2_=deserialize(params)
	W1g,b1g,W2g,b2g = sess.run([W1_grad,b1_grad,W2_grad,b2_grad], feed_dict={W1:W1_,b1:b1_,W2:W2_,b2:b2_,x:noisyPatches,expectedOutput:patches})
	return serialize(W1g,b1g,W2g,b2g)
'''

def objectiveAndGradient(params):
	W1_,b1_,W2_,b2_=deserialize(params)
	cost_,W1g,b1g,W2g,b2g = sess.run([cost,W1_grad,b1_grad,W2_grad,b2_grad], feed_dict={W1:W1_,b1:b1_,W2:W2_,b2:b2_,x:noisyPatches,expectedOutput:futurePatches})
	return cost_,serialize(W1g,b1g,W2g,b2g)

def assignToTensorFlow(params,counter):
	W1_,b1_,W2_,b2_=deserialize(params)
	sess.run([W1.assign(W1_),b1.assign(b1_),W2.assign(W2_),b2.assign(b2_),global_step_tensor.assign(counter)])

bfgsCounter=0
fig=plt.figure(figsize=(8, 8))

def callbackFromBFGS(params):	#this will be called every iteration
	global bfgsCounter
	bfgsCounter=bfgsCounter+1

	assignToTensorFlow(params,bfgsCounter)
	save_path = saver.save(sess, sessionFile)	#save sess	

	print bfgsCounter, sess.run([cost],feed_dict={x:noisyPatches,expectedOutput:futurePatches})

	if(bfgsCounter<=5 or bfgsCounter%25==0):
		global fig
		plotVectorField2D(autoencoderReconstruction,noisyPatches,imageFolder+"V%07d.png"%(bfgsCounter,),fig)
		plotDeformedGrid2D(autoencoderReconstruction, 0.1, patches,imageFolder+"G%07d.png"%(bfgsCounter,), fig)

	'''
	if bfgsCounter%10 == 0:
		assignToTensorFlow(params)

	if bfgsCounter%10 == 0:
		print(bfgsCounter, sess.run([cost],feed_dict={x:noisyPatches,expectedOutput:patches}))
	if (bfgsCounter%50==0 or bfgsCounter==1): 
		global fig
		plotVectorField2D(autoencoderReconstruction,noisyPatches,imageFolder+"V%07d.png"%(bfgsCounter,),fig)
		plotDeformedGrid2D(autoencoderReconstruction, 0.1, patches,imageFolder+"G%07d.png"%(bfgsCounter,), fig)
		#visualizeWeight(W_Grid,imageFolder+"W%07d.png"%(step,))
	if bfgsCounter%100==0:
		save_path = saver.save(sess, sessionFile)	#save sess	
	'''

if(runTraining):

	if(startNewTraining):
		sess.run(tf.initialize_all_variables())
	else:
		saver.restore(sess, sessionFile)
		bfgsCounter=sess.run(global_step_tensor)

	'''
	with tf.name_scope("cost"):
		tf.scalar_summary('total',cost)
		tf.scalar_summary('square_error',square_error)
		#tf.scalar_summary('contractive_penalty',contractive_penalty)
		#tf.scalar_summary('weight_decay',weight_decay)
		#tf.scalar_summary('sparsity_penalty',sparsity_penalty)
		#tf.scalar_summary('hidden_activation_rate',tf.reduce_mean(roh_hat))

	merged=tf.merge_all_summaries()
	writer=tf.train.SummaryWriter(logDirectory, sess.graph)

	current_global_step=tf.train.global_step(sess, global_step_tensor)
	'''
	#before training
	#print('beforeTrain:', sess.run([cost,square_error],feed_dict={x:patches}))

	'''
	fig=plt.figure(figsize=(8, 8))
	#actual run
	for step in range(current_global_step,current_global_step+100001):
		
		if ((stocasticTraining==True) and (step%roundPerTrainingSet==0)) or patches==None:
			#patches,targetPatches=sampleFromCircleFlow(0.75,10.0*np.pi/180.0,trainsetSize)
			#patches,not_used=sampleFromBean(trainsetSize)
			patches,not_used=sampleFromFunction(sTransform,trainsetSize)
			noisyPatches = addGaussianNoise(patches,0.1)

		summary,_ = sess.run([merged,train],feed_dict={x:noisyPatches,expectedOutput:patches})
		writer.add_summary(summary,step)

		if step%100 == 0:
			print(step, sess.run([cost,square_error],feed_dict={x:noisyPatches,expectedOutput:patches}))
		if (step<=2048 and (step in [2,4,8,16,32,64,128,256,512,1024,2048])) or (step%4000==0): 
			plotVectorField2D(autoencoderReconstruction,noisyPatches,imageFolder+"V%07d.png"%(step,),fig)
			plotDeformedGrid2D(autoencoderReconstruction, 0.1, patches,imageFolder+"G%07d.png"%(step,), fig)
			#visualizeWeight(W_Grid,imageFolder+"W%07d.png"%(step,))
		if step%10000==0:
			save_path = saver.save(sess, sessionFile)	#save sess	
	'''
	W1_now,b1_now,W2_now,b2_now=sess.run([W1,b1,W2,b2])
	#ret = minimize(objective, x0=serialize(W1_now,b1_now,W2_now,b2_now), method='BFGS', jac=gradient, callback=callbackFromBFGS, options={'maxiter':1000,'disp':True})
	ret = minimize(objectiveAndGradient, x0=serialize(W1_now,b1_now,W2_now,b2_now), method='BFGS', jac=True, callback=callbackFromBFGS, options={'maxiter':5000,'disp':True})
	#ret = minimize(objectiveAndGradient, x0=serialize(W1_now,b1_now,W2_now,b2_now), method='L-BFGS-B', jac=True, callback=callbackFromBFGS, options={'maxiter':5000,'maxcor':10,'disp':True}) #'maxls':20,

	save_path = saver.save(sess, sessionFile)	#save sess	

	plotVectorField2D(autoencoderReconstruction,noisyPatches,imageFolder+"V%07d.png"%(bfgsCounter,),fig,True)
	plotDeformedGrid2D(autoencoderReconstruction, 0.1, patches,imageFolder+"G%07d.png"%(bfgsCounter,), fig)