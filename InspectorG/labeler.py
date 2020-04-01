import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.metrics

RAND = None # default random seed

class CrossValidation():
    def __init__ (self, data, labels):
        """
        Args:
            data: data matrix
            labels: label for data
        """
        self.data = data
        self.labels = labels
        self.classes = set(labels)
        self.class_dslices = dict()
    
    def _divide_k_fold (self, data, labels, k):
        """
        Subfunction of slice_data_k_fold()
        Args:
            data: data matrix
            labels: label for data
            k: number of folds
        """
        fold, r = len(labels)//k, len(labels)%k
        data_slices = []

        cur_idx = 0
        for i in range(k):
            next_idx = cur_idx + fold + 1 if i < r else cur_idx + fold
            dslice = (list(data[cur_idx:next_idx]), list(labels[cur_idx:next_idx]))
            data_slices.append(dslice)
            cur_idx = next_idx
        return data_slices

    def slice_data_k_fold (self, k):
        """
        Slice the whole data into k folds
        """
        self.k = k
        for class_label in self.classes:
            sub_data = self.data[self.labels == class_label]
            sub_labels = np.ones(len(sub_data))*class_label

            dslices = self._divide_k_fold(sub_data, sub_labels, k)
            self.class_dslices[class_label] = dslices


    def choose_i_fold (self, i):
        """
        Choose ith fold of k folds and divide training data and validation data
        """
        xtmp, ytmp, xval, yval = [], [], [], []
        for fi in range(self.k):
            for cl in self.classes:
                (x, y) = self.class_dslices[cl][fi]
                if fi == i:
                    xval, yval = xval + x, yval + y
                else:
                    xtmp, ytmp = xtmp + x, ytmp + y            
        n = lambda l : np.array(l)
        return n(xtmp), n(ytmp), n(xval), n(yval)

def get_max_layer_size(tr_size):
    """
    maximum number of nodes for each layer    
    """
    for n in range(10):
        if 2**n > tr_size:
            break
    return n
    
def get_Perf_model_tmp(xtr, ytr, xte, yte, layer, iteration, rand):
    """
    Get performance of model that has input condition (layer structure, iteration, random seed, etc.)
    
    Args:
        xtr: training data matrix
        ytr: label for xtr
        xte: test data matrix
        yte: label for xte
        layer: structure of layers
        iteration:
        rand: random seed
    """
    layer_size = tuple(2**a for a in layer)
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=layer_size, random_state=rand, max_iter = iteration)
    model.fit(xtr, ytr)
    yscore = model.predict_proba(xte)[:, 1]
    ypred = model.predict(xte)
    if len(set(ytr)) == 2:
        f1 = sklearn.metrics.f1_score(yte, ypred)
    else:
        f1 = sklearn.metrics.f1_score(yte, ypred, average = 'macro')
    return model, f1 #average_precision

def search(func, first_layer, layer_num, min_layer, margin = 1):
    """
    Search for the best model in the whole search space.
       
    Args:
        func: function about input training method
        first_layer: input variable for recursive function
        layer_num: number of layers
        min_layer: minimum number of layers
        margin: margin for the searching range of the node number 
    """
    Hyper = []
    current = [0 for i in range(layer_num)]

    def construct_layer(prev_layer, depth):
        n = len(current) - depth 
        for i in range(min_layer, prev_layer+margin):
            current[n-1] = i
            if depth == 0:
                Hyper.append(func(current))
            else:
                construct_layer(i, depth-1)

    construct_layer(first_layer, layer_num-1)
    return Hyper

def make_function(xtr, ytr, iteration, rand):
    """
    make input train method function without cross validation
    """
    def get_ROC_F1(layer):
        layer = tuple(layer)
        _, f1 = get_Perf_model_tmp(xtr, ytr, xtr, ytr, layer, iteration, rand)
        return layer, f1
    return get_ROC_F1

def make_function_cv(xtr, ytr, iteration, k, rand):
    """
    make input train method function with cross validation
    """
    CV = CrossValidation(xtr, ytr)
    CV.slice_data_k_fold(k)
    def get_ROC_F1(layer):
        #print(layer)
        layer = tuple(layer)
        f1 = 0
        for i in range(k):
            xtmp, ytmp, xval, yval = CV.choose_i_fold(i)
            _, f1_tmp = get_Perf_model_tmp(xtmp, ytmp, xval, yval, layer, iteration, rand)
            f1 = f1+f1_tmp
        return layer, f1/k
    return get_ROC_F1

def train_recursive_cv(xtr, ytr, k, layernum, test_iter = 40, max_iter = 400, margin = 1, rand = RAND, min_layer = 2):
    """
    Search the possible mlp structures and return the best model
    
    Args:
        xtr: training data matrix
        ytr: label for xtr
        k: number of folds
        layernum: number of layers
        test_iter: number of test iterations
        max_iter: maximum number of iterations
        margin: margin for the searching range of the node number 
        rand: random seed
        min_layer: minimum number of layers
    """
    defectnum = sum(ytr)
    print(f'Number of Defective Data : {defectnum}, K (Cross Validation) : {k}')
     
    N = get_max_layer_size(np.shape(xtr)[1])
    layer_list, perf_list = [], []

    if k != 1:
        input_function = make_function_cv(xtr, ytr, max_iter, k, rand)
    else:
        input_function = make_function(xtr, ytr, test_iter, rand)
        
    candidates = search(input_function, N, layernum, min_layer, margin)
    print('Searching - complete')
    for layer, perf in candidates:
        layer_list.append(layer)
        perf_list.append(perf)

    idx = np.argmax(perf_list)
    max_model, _ = get_Perf_model_tmp(xtr, ytr, xtr, ytr, layer_list[idx], max_iter, rand)
    return max_model
