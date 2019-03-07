import numpy as np

class TreeNode(object):
    def __init__(self, feature_i=None, threshold=None, value=None, child=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.child = child

def divide_feature_index(X, feature_i, threshold):
    if isinstance(threshold.item(), int) or isinstance(threshold.item(), float):
        true_index = X[:,feature_i]>=threshold
        false_index = X[:,feature_i]<threshold
    else:
        true_index = X[:,feature_i]==threshold
        false_index = X[:,feature_i]!=threshold
    return true_index, false_index


class DecisionTree(object):
    def __init__(self, max_depth=float("inf"), min_samples_split=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._purity_calc = None
        self._leaf_calc_value = None
        self.feature_enable=None
        
    def fit(self, X, y, feature_enable=None):
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)        
        if feature_enable is None:
            self.feature_enable = np.arange(X.shape[1])
        else:
            self.feature_enable = feature_enable
            
        self.root = self._build_tree(X,y,0)
    
    def _build_tree(self, X, y, depth):
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)    
        m,n = X.shape
        label_value = np.unique(y)
        
        #只有1个类别，
        #if len(label_value)==1:
        #    return TreeNode(value=label_value[0])
        
        #最大深度、叶子节点上样本数量条件
        if len(label_value)==1 or depth==self.max_depth or m<=self.min_samples_split:
            return TreeNode(value=self._leaf_calc_value(y))

        max_purity = None
        for i in self.feature_enable:
            feature_value = np.unique(X[:,i])
            if len(feature_value)==1: pass
            for threshold in feature_value:
                true_index, false_index = divide_feature_index(X, i, threshold)
                if sum(true_index)>0 and sum(false_index)>0:
                    ytrue = y[true_index]
                    yfalse = y[false_index]

                    purity = self._purity_calc(y, X, true_index, false_index)
                    if max_purity is None or purity>max_purity:
                        max_purity = purity
                        decision = {"feature_index":i, "threshold":threshold}
                        divided = {"TrueX": X[true_index], "Truey":ytrue,
                                   "FalseX":X[false_index],"Falsey":yfalse}
        
        if max_purity is None:
            return TreeNode(value=self._leaf_calc_value(y))
        
        node = TreeNode(feature_i=decision["feature_index"], threshold=decision["threshold"])
        node.child = [self._build_tree(divided["TrueX"], divided["Truey"], depth+1),
                      self._build_tree(divided["FalseX"], divided["Falsey"],depth+1) 
                     ]
        return node
        
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        ptr = self.root
        while ptr.value is None :
            next_ptr = ptr.child[1]
            feature_value = x[ptr.feature_i]
            if isinstance(feature_value.item(), int) or isinstance(feature_value.item(), float):
                if feature_value >= ptr.threshold:
                    next_ptr = ptr.child[0]
            elif feature_value == ptr.threshold:
                next_ptr = ptr.child[0]
            ptr = next_ptr
        return ptr.value
    
def calculate_entropy(y):
    unique_labels = np.unique(y)
    entropy = 0
    for yi in unique_labels:
        count = sum(y == yi)
        p = count / len(y)
        entropy += -p * np.log2(p)
    return entropy

def calculate_gini(y):
    unique_labels = np.unique(y)
    gini = 1
    for yi in unique_labels:
        count = sum(y == yi)
        p = count / len(y)
        gini -= p**2
    return gini


class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, x, y1_index, y2_index):
        # Calculate information gain
        y1 = y[y1_index]
        y2 = y[y2_index]
        p = len(y1)/len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p*calculate_entropy(y1)-(1-p)*calculate_entropy(y2)
        info_gain_rate = info_gain/calculate_entropy(x)
        return info_gain_rate
    
    def _calculate_gini(self, y, x, y1_index, y2_index):
        y1 = y[y1_index]
        y2 = y[y2_index]
        p = len(y1)/len(y)
        Gyx = p*calculate_gini(y1) + (1-p)*calculate_gini(y2)
        return -Gyx
    
    def _most_vote(self, y):
        label_value = np.unique(y)
        return label_value[np.argmax([sum(y==yi) for yi in label_value])]

    def fit(self, X, y, feature_enable=None):
        #self._purity_calc = self._calculate_information_gain
        self._purity_calc = self._calculate_gini
        self._leaf_calc_value = self._most_vote
        super(ClassificationTree, self).fit(X, y, feature_enable=None)
        
class RegressionTree(DecisionTree):
    def _calculate_square_error(self, y, x, y1_index, y2_index):
        #print(y1,y2)
        y1 = y[y1_index]
        y2 = y[y2_index]
        S = np.std(y1)**2*len(y1) + np.std(y2)**2*len(y2)
        
        return -S
    
    def _mean_value(self, y):
        return np.mean(y)

    def fit(self, X, y, feature_enable=None):
        self._purity_calc = self._calculate_square_error
        self._leaf_calc_value = self._mean_value
        super(RegressionTree, self).fit(X, y, feature_enable=None)