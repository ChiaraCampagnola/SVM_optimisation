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
        alpha = alpha.reshape(-1,1)
        
        # Check dimensions
        assert alpha.shape[1] == 1
        assert self.Q.shape[0] == alpha.shape[0], f'Wrong dimensions, Q = {self.Q.shape}, alpha: {alpha.shape}'
        assert self.Q.shape[1] == alpha.shape[0], f'Wrong dimensions, Q = {self.Q.shape}, alpha: {alpha.shape}'
        
        #print(f'Q: {self.Q}')
        #print(f'alpha: {alpha}')
        #print(f' Zero: {alpha[alpha==0]}')
        #print(f'matmul: {np.transpose(alpha) @ self.Q @ alpha}')
        
        # Standard SVM dual function, g
        g = np.sum(alpha) - 0.5 * np.transpose(alpha) @ self.Q @ alpha
        #print(f'g: {g}')
        # Barriers
        barriers = - np.sum(np.log(alpha)) - np.sum(np.log(self.C-alpha))
        
        return -self.t*g + barriers
    
    def df(self, alpha):
        '''
        Returns gradient of the objective function
        '''
        dg = 1 - self.Q @ alpha # Gradient of g
        dbarriers = 1/(self.C - alpha) - 1/alpha # Gradient of the barriers
        
        assert np.isnan(np.sum(dbarriers)) == False, 'NaN values in dbarriers'
        
        return -self.t*dg + dbarriers
    
    def d2f(self, alpha):
        '''
        Return hessian of the objective function
        '''
        
        d2f = self.t*self.Q + np.diag(1/np.power(alpha,2)) + np.diag(1/np.power(self.C-alpha,2))
        
        assert np.isnan(np.sum(d2f)) == False, 'NaN values in d2f'
        
        return d2f