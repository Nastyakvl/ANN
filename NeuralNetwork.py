# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:44:16 2019

@author: Nastya
"""
import numpy.random as r
from time import time
import numpy as np
from sklearn import  metrics
import pylab as p
from sklearn.preprocessing import OneHotEncoder


# sigmoid 
def sigmoid(x,deriv=False):
    if(deriv==True):
        return sigmoid(x)*(1-sigmoid(x))
    return 1/(1+np.exp(-x))

#def softmax(x):
#    return np.exp(x) / float(sum(x))

class NeuralNetwork:
    def __init__(self, nn_structure, nu, enc):
        self.nn_structure = nn_structure  # structure of ANN
        self.nu = nu
        self.W,self.b = self.setup_and_init_weights()
        self.enc = enc  # encoder for outputs
        
    #initialize weights
    def setup_and_init_weights(self):
        W = {}
        b = {}
        for l in range(1, len(self.nn_structure)):
            W[l] = r.random_sample((self.nn_structure[l], self.nn_structure[l-1]))
            b[l] = r.random_sample((self.nn_structure[l],1))
        return W, b


    def feed_forward(self,x):
        z={}
        o={}
        for l in range(1, len(self.W)+1):
        #if it's a first layer then use x
        #otherwise output from previous layer
            if l == 1:
                node_in = x
            else:
                node_in = o[l-1]
        
            z[l] = (self.W[l].dot(node_in)) + self.b[l] # z^(l+1) = W^(l)*h^(l) +b[l] 
            
            o[l] = sigmoid(z[l]) # h^(l) = f(z^(l)) 
        
        
        return o

    def back_forward(self, x,t, o,):
        delta={}
        deltaW={}
        deltaB={}
        for l in range(len(self.W),0,-1):
                if (l==len(self.W)):
            
                    #output layer
                    delta[l] = o[l]*(1-o[l])*(t-o[l])
                    deltaW[l]=self.nu*delta[l].dot((o[l-1]).T)
                    deltaB[l]=self.nu*delta[l]
           
        
                #hidden layer
                else:
                    if(l==1):
                        node_in=x
                    else:
                        node_in=o[l-1].T
            
                    delta[l] = o[l]*(1-o[l])*(self.W[l+1].T).dot(delta[l+1])
                    deltaW[l]=self.nu*delta[l].dot(node_in)
                    deltaB[l]=self.nu*delta[l]
                          
            
        return deltaW,deltaB
    
    
    def train(self,X_train,y_train, X_valid,y_valid):
           accuracy=0
           bestW=self.W
           bestB=self.b
           accuracyAll=[]
           summW={}
           summB={}
           alpha=0.8
           bestAccuracy=0

           #training. 40 times
           m=0
           while(m<40):
               

# update if stuck
#               if(time()-start2>180):
#                   start2=time()
#                   self.W,self.b=self.setup_and_init_weights()
#                   m = 0
#                   accuracyAll=[]
#                   print("update")
           
               for i in range(0, int((np.shape(X_train)[0])/10)):
           
                   for l in range(len(self.W),0,-1):
                       summW[l]=0
                       summB[l]=0
                   
                   #train on 10 examples and then update weights
                   for n in range(10*i,10*i+10):
                       #feed_forward
                       forw=self.feed_forward(X_train[n].reshape(len(X_train[n]),1))
                       
                       #back forward
                       deltaW,deltaB = self.back_forward(X_train[n].reshape(1,len(X_train[n])),y_train[n].reshape(len(y_train[0]),1),forw) #0.3
                       
                       #calcuate all delta
                       for l in range(len(self.W),0,-1):
                           summW[l]+=deltaW[l]
                           summB[l]+=deltaB[l]
                   
                    
                    #update weights        
                   for l in range(len(self.W),0,-1):
                       self.W[l]+=alpha*summW[l]
                       self.b[l]+=alpha*summB[l]
        
               #validation
               predictedV=np.full([len(y_valid)],0, dtype=float) 
               
               for i in range(0, np.shape(X_valid)[0]):
                   o=self.feed_forward(X_valid[i].reshape(len(X_valid[i]),1))
                   
                   #output from ANN
                   predictedV[i]=self.enc.inverse_transform(o[len(self.nn_structure)-1].reshape(1,len(o[len(self.nn_structure)-1])))
                   
           
               accuracy=metrics.accuracy_score(y_valid.reshape(len(y_valid)), predictedV.round())
               accuracyAll.append(accuracy)
               
               # save the best state of ANN
               if(accuracy>bestAccuracy):
                   print(m)
                   bestW=self.W
                   bestB=self.b
                   bestAccuracy=accuracy
                   
                   
               self.W=bestW
               self.b=bestB
               
               m+=1
      
       
# plot the accuracy on validation set  
           p.plot(range(0,m),accuracyAll, label='accuracy')
           p.ylabel('Accuracy')
           p.xlabel('iteration')
           p.show()
           
           
               
               
    def predict(self,X_test):
        predicted=[]

        #testing
        for i in range(0, np.shape(X_test)[0]):
           o=self.feed_forward(X_test[i].reshape(len(X_test[i]),1))
           predict = self.enc.inverse_transform(o[len(self.nn_structure)-1].reshape(1,len(o[len(self.nn_structure)-1])))
           predicted.append(predict)
           
        arr=np.array(predicted)
        return np.array(arr.reshape(len(predicted),1))
        