# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:17:38 2019

@author: Nastya
"""

from NeuralNetwork import NeuralNetwork
import numpy as np
import csv
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder

def readData(file_Name):
    X=np.full([42000, 784],0, dtype=float)
    Y=np.full([42000,1],0, dtype=float)

    i=0
    
    with open(file_Name, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if i == 0:
                i+=1
                continue
            
            x=list(map(int, row))
            Y[i-1] = x[0] 
            x=np.delete(x,0)
            X[i-1] = x
            i+=1
            

    Y=Y.reshape(len(Y),1) 

    return X,Y     
    

def main():
    X,Y = readData('train.csv')
    

    
    #scale the data
    X_scale = StandardScaler()
    X = X_scale.fit_transform(X)
   
    #split test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
  
    #split traind and validation data
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.122)
    
    data = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
    data = np.array(data)
    enc = OneHotEncoder(sparse = False)
    y_train = enc.fit_transform(y_train.reshape(len(y_train),1))
  
    #structure of ANN 
    nn_structure=[784,50,20,10]
    
    #create the ANN
    NN=NeuralNetwork(nn_structure,0.1, enc)
    
    #train the ANN
    NN.train(X_train,y_train,X_valid,y_valid)
    
    #presict the result with trained ANN
    predicted=NN.predict(X_test)
    print("Accuracy: %f\n" % metrics.accuracy_score(y_test.reshape(len(y_test)), predicted.round()))
 
   
    cm = metrics.confusion_matrix(y_test.reshape(len(y_test)), predicted.round())
    print("\nConfusion matrix:\n")
    print(cm)

    cm_norm = normalize(cm.astype(np.float64), axis=1, norm='l1')
    print("\nNormalized confusion matrix:\n")
    print(cm_norm)
   
    print("Classification report:\n%s\n"
     % metrics.classification_report(y_test.reshape(len(y_test)), predicted.round()))
 
  
    
            
            
if __name__ == "__main__":
    main()
