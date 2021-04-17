import numpy as np
import numexpr as ne
from sklearn import svm as sk_svm

def rbf_kernel(train_X, test_X):
    '''
    [Slightly faster than scikit]

    Calculate the NxM RBF kernel matrix (N = # training examples, M = # test examples)
    K(i,j) = exp(-gamma*||i-j||^2)
    Note: ||i-j||^2 = ||i||^2 + ||j||^2 -2<i,j> = A + B - 2*C

    Inputs:
    train_X: array (N, #features)
    test_X: array (M, #features)
    '''
    gamma=1/train_X.shape[1]
    
    if train_X.ndim == 1 or test_X.ndim == 1:
        print('ERROR: train_X input to rbf kernel has wrong shape')
        return

    A = np.linalg.norm(train_X, axis=1)**2
    B = np.linalg.norm(test_X, axis=1)**2
    C = train_X @ np.transpose(test_X)

    K = ne.evaluate('exp(-g * (A + B - 2 * C))', {
        'A' : A[:,None],
        'B' : B[None,:],
        'C' : C,
        'g' : gamma,
        })
    
    return K

def scikit_svm(kernel, train_X, train_y, test_X, test_y):
    
    if kernel == 'rbf':
        svm = sk_svm.SVC(kernel='rbf', C = 1.0)
        
    if kernel == 'linear':
        svm = sk_svm.SVC(kernel='linear', C = 1.0)
        
    svm.fit(train_X, train_y)
    pred_y = svm.predict(test_X)
        
    return accuracy(pred_y, test_y)
    
def accuracy(training, test):
    total = len(training)
    correct = np.sum(training==test)
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

def calculate_g(alpha, Q):
    A = np.sum(alpha)
    a, b = np.meshgrid(alpha,alpha)
    B = np.multiply(a,b)
    B = np.multiply(B, Q)
    B = np.sum(B)
    
    return A-0.5*B