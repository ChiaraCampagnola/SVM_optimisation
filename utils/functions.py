import numpy as np

from utils.utils import Q_matrix

class ObjectiveFunction:
    '''
    Objective function for the barrier method
    '''
    def __init__(self, train_X, train_y, kernel):
        self.train_X = train_X
        self.train_y = train_y
        self.kernel = kernel
        self.Q = Q_matrix(train_X, train_y, kernel)
    
    def f(self, alpha, C, t):
        '''
        The dual SVM function + the log barriers
        
        
        '''
        
        # Standard SVM dual function, g
        a, b = np.meshgrid(alpha,alpha)
        B = np.multiply(a,b)
        B = np.sum(np.multiply(B, self.Q))
    
        g = np.sum(alpha) - 0.5 * B
        
        # Barriers
        barriers = - np.sum(np.log(alpha)) - np.sum(np.log(C-alpha))
        
        return -t*g + barriers