import numpy as np
from optimisation.newton import newton_method
from optimisation.backtracking import backtracking
from utils.functions import ObjectiveFunction

class BarrierMethod:
    def __init__(self, tol, kernel_fun, train_X, train_y, C):
        self.tol = tol
        self.kernel_fun = kernel_fun
        self.train_X = train_X
        self.train_y = train_y.reshape(-1,1)
        self.C = C
        
        self.num_examples = len(self.train_y)
        self.kernel = self.kernel_fun(self.train_X, self.train_X)
        self.alpha = np.zeros(self.num_examples).reshape(-1,1)
        self.A = np.transpose(self.train_y)
        self.b = 0
        self.g_iterates = []
        
        self.m = 2*self.num_examples # Number of inequality constraint (since 0 < alpha < C we have 2 for each alpha)
        self.t = self.m/self.tol # Check this
        self.mu = 1.25
        self.F = ObjectiveFunction(self.train_X, self.train_y, self.kernel_fun, self.C, self.t)
        
    def fit(self):
        
        # Set feasible starting point
        self.set_feasible_starting_point()
        
        # Repeat
        while self.m/self.t >= self.tol:
            # Minimize fun and update alpha
            self.alpha, _ = newton_method(self.F, self.alpha, self.A, self.b, backtracking)
        
            # Increase t
            self.t = self.mu * self.t
            
        # Find support vectors so we keep only elements corresponding to suport vectors
        SV = self.alpha.squeeze() > 0
        
        result = {'train_X': self.train_X[SV,:],
                  'train_y': self.train_y[SV],
                  'alpha': self.alpha[SV],
                  'b': self.b,
                  'g': self.g_iterates}
        
        return result
        
    def set_feasible_starting_point(self):
        '''
        Note: breaks if all the examples are either positive or negative
        TO DO: add safeguard
        '''
        
        # Count positive and negative examples
        pos = np.sum(self.train_y == 1)
        neg = np.sum(self.train_y == -1)
        
        # if pos > neg:
        #     self.alpha[self.train_y == -1] = self.C
        #     self.alpha[self.train_y == 1] = self.C * neg/pos
        # else:
        #     self.alpha[self.train_y == 1] = self.C
        #     self.alpha[self.train_y == -1] = self.C * pos/neg
        
        self.alpha[self.train_y == 1] = self.C/pos
        self.alpha[self.train_y == -1] = self.C/neg
        
        # Check it does break any conditions
        assert np.isnan(np.sum(self.alpha)) == False, 'NaN values in alpha'
        np.testing.assert_almost_equal(np.dot(np.transpose(self.alpha), self.train_y), 0, decimal=10)
        assert np.any(self.alpha > self.C) == False, 'alpha higher than C'
        assert np.any(self.alpha < 0) == False, 'alpha lower than 0'