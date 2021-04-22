import numpy as np
import numexpr as ne
from sklearn import svm as sk_svm

def scikit_svm(kernel, train_X, train_y, test_X, test_y):
    
    if kernel == 'rbf':
        svm = sk_svm.SVC(kernel='rbf', C = 1.0)
        
    if kernel == 'linear':
        svm = sk_svm.SVC(kernel='linear', C = 1.0)
        
    svm.fit(train_X, train_y)
    pred_y = svm.predict(test_X)
        
    return accuracy(pred_y, test_y)
    
def accuracy(training, test):
    training = training.squeeze()
    test = test.squeeze()
    
    total = len(training)
    correct = np.sum(training==test)
    
    #print(total)
    #print(correct)
    return correct/total

def predict(alpha, kernel_matrix, b, train_y):
    return np.sign(np.multiply(alpha, train_y) @ kernel_matrix + b)

def Q_matrix(train_X, train_y, kernel):
    a, b = np.meshgrid(train_y,train_y)
    yy = np.multiply(a,b)
    k = kernel(train_X, train_X)
    
    return np.multiply(yy,k)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def calculate_g(alpha, train_X, train_y, kernel):
    alpha_y = np.multiply(alpha, train_y)
    return np.sum(alpha) - 0.5*np.transpose(xy)@kernel@xy