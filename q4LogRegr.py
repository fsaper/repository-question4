# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 14:27:38 2015

@author: Fernando Sa-Pereira 119017503
"""

def shuffleData(X,y):
    import random
    assert len(X) == len(y)
    shuffX = []
    shuffY = []
    order = range(len(X))
    random.shuffle(order)
    for i in range(len(X)):
        shuffX.append(X[order[i]])
        shuffY.append(y[order[i]])
    return shuffX,shuffY

def splitIntoSets(X,y,k,fold):
    assert fold >=0
    assert fold < k    
    size = len(X)/k
    testPts = range(fold*size,(fold+1)*size)
    trainX = [X[i] for i in range(len(X)) if i not in testPts]
    trainY = [y[i] for i in range(len(y)) if i not in testPts]
    testX =  X[fold*size:(fold+1)*size]
    testY =  y[fold*size:(fold+1)*size]
            
    return trainX, trainY, testX, testY

def classStats(X,y,classList):
    XbyClass = []
    means = []
    covar = []       
    #split X by class    
    for c in classList:
        XbyClass.append([])
        
    for i in range(len(X)):  
        XbyClass[classList.index(y[i])].append(X[i])
    
    for i in range(len(classList)):
        for _ in range(len(X[0])): # i.e., for each feature in X
            means.append(np.mean(XbyClass[i],axis=0))
        print i, means[i]
            
    return means,covar
    
if __name__ == "__main__":   
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    k = 5
    X = np.loadtxt("wpbcx.dat")
    y = np.loadtxt("wpbcy.dat")
    classList = list(set(y)) #create a list of all classes
    X,y = shuffleData(X,y)
    totScore=0
    totTrials=0
    for fold in range(k):
        #print "fold: ",fold
        trainX, trainY, testX, testY = splitIntoSets(X,y,k,fold)
        """
        logistic regression
        """
        logreg = LogisticRegression()
        logreg.fit(trainX,trainY)
        predictions = logreg.predict(testX)
        logLikli = logreg.predict_log_proba(testX)  
        #print logLikli
        results = abs(testY - predictions)
        score = logreg.score(testX,testY)
        totScore += score
        
        """
        quadratic discriminant analysis, diagonal covariance
        """
        means,covar = classStats(X,y,classList)   
        
    #print totScore/float(k),np.mean(logLikli)
    
    
               