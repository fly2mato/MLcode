import numpy as np
from tqdm import tqdm
from DecisionTrees import *

class RandomForest(object):
    def __init__(self, n_estimators=100, max_depth=float("inf"), 
                 min_samples_split=2, max_features=None):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.max_features=max_features
        self.forest = []
        for i in range(self.n_estimators):
            tree = ClassificationTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            self.forest.append(tree)
        
    def fit(self, X, y):
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)
        samples_index = np.arange(X.shape[0])
        feature_index = np.arange(X.shape[1])
        
        for i in tqdm(range(self.n_estimators)):
            x_index = np.random.choice(samples_index, size=X.shape[0], replace=True)
            if self.max_features is None or self.max_features>=X.shape[1]:
                f_index = feature_index
            else:
                f_index = np.random.choice(feature_index, size=self.max_features, replace=False)
            self.forest[i].fit(X[x_index], y[x_index], f_index)           
    
    def _predict(self, x):
        y = [tree._predict(x) for tree in self.forest]
        label_value = np.unique(y)
        return label_value[np.argmax([sum(y==yi) for yi in label_value])]
        
    
    def predict(self, X):
        return [self._predict(x) for x in X]