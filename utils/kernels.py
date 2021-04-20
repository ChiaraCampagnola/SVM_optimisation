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

def linear_kernel(X_train, X_test):
    return np.matmul(X_train, np.transpose(X_test))

def polynomial_kernel(X_train, X_test, degree = 2, gamma = None, c = 1):
    
    if gamma is None:
        gamma = 1/X_train.shape[1] # Default value = 1/number of features
    
    dot = np.matmul(X_train, np.transpose(X_test))
    
    return np.power(gamma*dot + c, degree)