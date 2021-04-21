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
        g = np.sum(alpha) - 0.5 * np.transpose(alpha) @ self.Q @ alpha
        
        # Barriers
        barriers = - np.sum(np.log(alpha)) - np.sum(np.log(C-alpha))
        
        return -t*g + barriers
    
    def df(self, alpha, C, t):
        '''
        Returns gradient of the objective function
        '''
        dg = 1 - self.Q @ alpha # Gradient of g
        dbarriers = 1/(C - alpha) - 1/alpha # Gradient of the barriers
        
        return -t*dg + dbarriers