import numpy as np
from DecisionTrees import *
from tqdm import tqdm

class SquareError(object):
    def loss(self, y, y_pred):
        return (y-y_pred).dot(y-y_pred)
    
    def gradient(self, y, y_pred):
        return -2*(y-y_pred)
    
    def hessian(self, y, y_pred):
        return 2*np.ones(y.shape)

class CrossEntropy(object):
    def gradient(self, y, y_pred):
        return -y + 1/(1+np.exp(-y_pred))
    
    def hessian(self, y, y_pred):
        z = 1/(1+np.exp(-y_pred))
        return z*(1-z)
    
class xgboostTree(DecisionTree):
    def __init__(self, max_depth=float("inf"), min_samples_split=1, loss=None, lamb=0, gamma=0):
        self.loss = loss()
        self.lamb = lamb
        self.gamma = gamma
        super(xgboostTree, self).__init__(max_depth=max_depth, min_samples_split=min_samples_split)
    
    def _calculate_loss_gain(self, y, x, true_index, false_index):
        GL = sum(y[true_index,-2])
        GR = sum(y[false_index,-2])
        HL = sum(y[true_index,-1])
        HR = sum(y[false_index,-1])
        return .5*(GL**2/(HL+self.lamb) + GR**2/(HR+self.lamb) - (GL+GR)**2/(HL+HR+self.lamb))-self.gamma
    
    def _best_value(self, y):
        G = sum(y[:,-2])
        H = sum(y[:,-1])
        return -G/H+self.lamb

    def fit(self, X, y, y_pred_last, feature_enable=None):
        self._purity_calc = self._calculate_loss_gain
        self._leaf_calc_value = self._best_value
        G=self.loss.gradient(y, y_pred_last).reshape(-1,1)
        H=self.loss.hessian(y, y_pred_last).reshape(-1,1)
        if feature_enable is None: 
            feature_enable = np.arange(X.shape[1])
        yGH = np.hstack([y.reshape(-1,1), G, H])
        super(xgboostTree, self).fit(X, yGH, feature_enable=feature_enable)
        
    def predict(self, X):
        y = super(xgboostTree, self).predict(X)
        return y

class xgboost(object):
    def __init__(self, max_depth=float("inf"), min_samples_split=2, n_estimators=1, 
                learning_rate=1, loss=None, base_estimator=xgboostTree):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.base_estimator=base_estimator
        self.trees=[]
        for i in range(self.n_estimators):
            self.trees.append(self.base_estimator(max_depth, min_samples_split, loss=loss))
    
    def fit(self, X, y):
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)
        y_pred = np.zeros(y.shape)
        for tree in tqdm(self.trees):
            tree.fit(X,y,y_pred)
            y_pred += self.learning_rate*tree.predict(X)
            
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred