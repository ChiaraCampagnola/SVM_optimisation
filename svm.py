import numpy as np
import matplotlib.pyplot as plt

from utils.utils import accuracy
from utils.kernels import *
from optimisation.smo import SMO

class SVM:
    def __init__(self,
                 optim = 'SMO',
                 kernel = 'rbf',
                 C = 1,
                 max_passes = 5,
                 fit_tol = 1e-3,
                 calc_g_iterates = False):
        
        # Check that inputs are valid
        if optim not in ['SMO', 'SMO']:
            print("ERROR: optimisation method not supported.")
            return
        if kernel not in ['rbf', 'linear', 'polynomial']:
            print("ERROR: kernel type not supported.")
            return
        
        
        # Init from settings
        self.optim = optim
        self.C = C
        self.kernel_type = kernel
        self.tol = fit_tol
        self.max_passes = max_passes
        self.calc_g_iterates = calc_g_iterates
        
        if self.kernel_type == 'rbf':
            self.kernel = rbf_kernel
        elif self.kernel_type == 'linear':
            self.kernel = linear_kernel
        elif self.kernel_type == 'polynomial':
            self.kernel = polynomial_kernel
        
        self.has_been_fit = False # Flag to see if the fit method has been called
        
    def fit(self, train_X, train_y):
        '''
        Fit the SVM on the training data
        '''
        
        
        if self.optim == 'SMO':
            smo = SMO(tol = self.tol,
                      C = self.C,
                      kernel_fun=self.kernel,
                      train_X = train_X,
                      train_y = train_y,
                      max_passes = self.max_passes,
                      calc_g_iterates = self.calc_g_iterates)
            result = smo.fit()
            
        
        # Overwrite training data to exclude non support vectors
        self.train_X = result['train_X']
        self.train_y = result['train_y'] 
        
        # Save prediction parameters
        self.b = result['b']
        self.alpha = result['alpha']
        
        # Save the dual function iterations (to analyse convergence)
        self.g_iterates = result['g']
        
        self.has_been_fit = True
    
    def predict(self, test_X):
        '''
        Predict on test data
        '''
        kernel_matrix = self.kernel(self.train_X, test_X)
        if not self.has_been_fit:
            print("ERROR: the SVM has not been fit with training data.")
            
        return np.sign(np.multiply(self.alpha, self.train_y) @ kernel_matrix + self.b)  

    def get_accuracy(self, test_X, test_y):
        '''
        Predict on test data and output accuracy
        '''
        if not self.has_been_fit:
            print("ERROR: the SVM has not been fit with training data.")
        
        pred_y = self.predict(test_X)
        return accuracy(pred_y, test_y)
    
    def plot_convergence(self):
        '''
        Plot:
            - the error of g vs iteration number 
            - the convergence of g (g_{t+1} - g_star)/(g_t - g_star)
        '''
        g_star = self.g_iterates[-1] # Use the last value of g obtained as the comparison
        g_min_gstar = self.g_iterates-g_star
        nominator = g_min_gstar[1:] # g_{t+1} - g_star
        denominator = g_min_gstar[:-1] # g_t - g_star
        iterates = np.divide(nominator, denominator)
        t = np.arange(1,len(iterates)+1)
        
        # Plot
        plt.figure()
        plt.plot(t, iterates, label='$(g_{t+1}-g^*)/(g_t-g^*)$')
        plt.plot(t, -g_min_gstar[:-1], label='Error $(g_t-g^*)$')
        plt.legend(fontsize = 'large')
        plt.show()