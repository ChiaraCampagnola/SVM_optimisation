import numpy as np

from utils.utils import Q_matrix

class ObjectiveFunction:
    '''
    Objective function for the barrier method
    '''
    def __init__(self, train_X, train_y, kernel, C, t):
        self.train_X = train_X
        self.train_y = train_y
        self.kernel = kernel
        self.Q = Q_matrix(train_X, train_y, kernel)
        self.C = C
        self.t = t
    
    def f(self, alpha):
        '''
        The dual SVM function + the log barriers
        '''
        # Standard SVM dual function, g
        g = np.sum(alpha) - 0.5 * np.transpose(alpha) @ self.Q @ alpha
        
        # Barriers
        barriers = - np.sum(np.log(alpha)) - np.sum(np.log(self.C-alpha))
        
        return -self.t*g + barriers
    
    def df(self, alpha):
        '''
        Returns gradient of the objective function
        '''
        dg = 1 - self.Q @ alpha # Gradient of g
        dbarriers = 1/(self.C - alpha) - 1/alpha # Gradient of the barriers
        
        return -self.t*dg + dbarriers