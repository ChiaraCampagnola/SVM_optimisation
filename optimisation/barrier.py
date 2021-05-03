import numpy as np
from optimisation.newton import newton_method
from optimisation.backtracking import backtracking
from utils.functions import ObjectiveFunction

class BarrierMethod:
    def __init__(self, tol, kernel_fun, train_X, train_y, C, mu, t0):
        self.tol = tol
        self.kernel_fun = kernel_fun
        self.train_X = train_X
        self.train_y = train_y
        self.C = C
        
        self.num_examples = len(self.train_y)
        self.kernel = self.kernel_fun(self.train_X, self.train_X)
        self.alpha = np.zeros(self.num_examples)
        self.A = np.transpose(self.train_y)
        self.b = 0
        
        self.alpha_iterates = [self.alpha]
        self.alpha_0 = self.alpha <= 0
        
        self.m = 2*self.num_examples # Number of inequality constraint (since 0 < alpha < C we have 2 for each alpha)
        #self.t = self.m/self.tol # Check this
        self.t = t0
        if mu is None:
            self.mu = 1.25
        else:
            self.mu = mu
            
        self.F = ObjectiveFunction(self.train_X, self.train_y, self.kernel_fun, self.C, self.t)
        
        
    def fit(self):
        
        # Set feasible starting point
        self.set_feasible_starting_point()
        
        self.duality_gaps_theoretical = []
        self.f_iterates = [self.F.f(self.alpha)]
        self.newton_iterations = []
        
        max_iter = 100
        i = 0
        
        prev_alpha = self.alpha
        
        # Repeat
        while self.m/self.t >= self.tol and i < max_iter:
            
            
            i += 1
            # Minimize fun and update alpha
            
            self.alpha, nIter = newton_method(self.F, self.alpha, self.A, self.C, backtracking)
            #print(f'Newton iterations: {nIter}')
            
            self.alpha_iterates.append(self.alpha.squeeze())
            self.duality_gaps_theoretical.append(self.m/self.t)
            self.f_iterates.append(self.F.f(self.alpha))
            self.newton_iterations.append(nIter)
            
            # Increase t
            self.t = self.mu * self.t
            prev_alpha = self.alpha
            
            
        self.b = np.mean(self.train_y - (self.alpha*self.train_y)@self.kernel) 
        
        result = {'train_X': self.train_X,
                  'train_y': self.train_y,
                  'alpha': self.alpha,
                  'b': self.b,
                  'alpha_iter': self.alpha_iterates,
                  'theory_duality_gaps': self.duality_gaps_theoretical,
                  'f_iterates': self.f_iterates,
                 'newton_iter': self.newton_iterations}
        
        return result
        
    def set_feasible_starting_point(self):
        '''
        Note: breaks if all the examples are either positive or negative
        TO DO: add safeguard
        '''
        
        # Count positive and negative examples
        pos = np.sum(self.train_y == 1)
        neg = np.sum(self.train_y == -1)
        
        self.alpha[self.train_y == 1] = self.C/pos
        self.alpha[self.train_y == -1] = self.C/neg
        
        # Check it does break any conditions
        assert np.isnan(np.sum(self.alpha)) == False, 'NaN values in alpha'
        np.testing.assert_almost_equal(np.dot(np.transpose(self.alpha), self.train_y), 0, decimal=10)
        assert np.any(self.alpha > self.C) == False, 'alpha higher than C'
        assert np.any(self.alpha < 0) == False, 'alpha lower than 0'
        
#     def remove_non_SV(self):
        
#         SV = self.alpha.squeeze() > 0
#         self.alpha = self.alpha[SV]
#         self.train_y = self.train_y[SV]
#         self.train_X = self.train_X[SV,:]
#         self.A = self.train_y.reshape(1,-1)
#         self.F.reset(self.train_X, self.train_y)
#         self.kernel = self.kernel_fun(self.train_X, self.train_X)
        
    def predict(self, test_X):
        '''
        Predict on test data
        '''
        kernel_matrix = self.kernel_fun(self.train_X, test_X)
        
        return np.sign(np.transpose(np.multiply(self.alpha, self.train_y)) @ kernel_matrix + self.b)  

    def get_accuracy(self, test_X, test_y):
        '''
        Predict on test data and output accuracy
        '''
        
        pred_y = self.predict(test_X)
        return self.accuracy(pred_y, test_y)
    
    def accuracy(self, training, test):
        training = training.squeeze()
        test = test.squeeze()
        
        total = len(training)
        correct = np.sum(training==test)
        
        return correct/total