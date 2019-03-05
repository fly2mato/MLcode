import numpy as np
import copy
from tqdm import tqdm

from DecisionTrees import *

class AdaBoost(object):
    def __init__(self, base_estimator=ClassificationTree(max_depth=3, min_samples_split=2),
                n_estimators=50):
        self.n_estimators=n_estimators
        #self.learning_rate = learning_rate
        self.alpha = []
        self.base_estimators = []
        for i in range(n_estimators):
            self.base_estimators.append(copy.copy(base_estimator))            
        
    def fit(self, X, y):
        #y=1,-1
        
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1) 
        self.ylabel = np.unique(y)        
        m,n = X.shape
        W = np.ones(y.size)/m
        sample_index = np.array(X.shape[0])
        for i in tqdm(range(self.n_estimators)):
            x_index = np.random.choice(sample_index, size=m, p=W, replace=True)
            self.base_estimators[i].fit(X[x_index],y[x_index])
            y_pred = self.base_estimators[i].predict(X[x_index])
            index = y_pred != y[x_index]
            error = sum(W[index])
            am = .5*np.log((1-error)/error)
            self.alpha.append(am)
            
            
            origin_false_index = np.unique(x_index[index])
            origin_true_index = np.unique(x_index[~index])
            
            #right samples
            W[origin_true_index] *= np.exp(-am)
            #wrong samples
            W[origin_false_index] *= np.exp(am)
            
            W = W/sum(W)
    
    def predict(self, X):
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1) 
        y_pred = np.zeros(X.shape[0])        
        for i in range(self.n_estimators):
            y = self.base_estimators[i].predict(X)
            y_pred += (np.ones(X.shape[0])*(y==self.ylabel[1])*2-1)*self.alpha[i]
        return np.sign(y_pred)