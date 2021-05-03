import numpy as np
import matplotlib.pyplot as plt

from utils.utils import accuracy
from utils.kernels import *
from optimisation.smo import SMO
from optimisation.barrier import BarrierMethod

class SVM:
    def __init__(self,
                 optim = 'SMO',
                 kernel = 'rbf',
                 C = 1,
                 max_passes = 5,
                 fit_tol = 1e-3,
                 calc_g_iterates = False,
                 mu = None,
                 t = None):
        
        # Check that inputs are valid
        if optim not in ['SMO', 'barrier']:
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
        self.mu = mu
        self.t = t
        
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
            
        if self.optim == 'barrier':
            barrier = BarrierMethod(tol = self.tol,
                                    kernel_fun = self.kernel,
                                    train_X = train_X,
                                    train_y = train_y,
                                    C = self.C,
                                    mu = self.mu,
                                    t0 = self.t)
            result = barrier.fit()
        
        # Overwrite training data to exclude non support vectors
        self.train_X = result['train_X']
        self.train_y = result['train_y'] 
        
        # Save prediction parameters
        self.b = result['b']
        self.alpha = result['alpha']
        
        # Save the dual function iterations (to analyse convergence)
        if self.optim == 'smo':
            self.g_iterates = result['g']
        if self.optim == 'barrier':
            self.alpha_iter = result['alpha_iter']
            self.duality_gaps_theoretical = result['theory_duality_gaps']
            self.f_iterates = result['f_iterates']
            self.newton_iter = result['newton_iter']
        
        self.has_been_fit = True
    
    def predict(self, test_X):
        '''
        Predict on test data
        '''
        kernel_matrix = self.kernel(self.train_X, test_X)
        if not self.has_been_fit:
            print("ERROR: the SVM has not been fit with training data.")
        
        return np.sign(np.transpose(np.multiply(self.alpha, self.train_y)) @ kernel_matrix + self.b)  

    def get_accuracy(self, test_X, test_y):
        '''
        Predict on test data and output accuracy
        '''
        if not self.has_been_fit:
            print("ERROR: the SVM has not been fit with training data.")
        
        pred_y = self.predict(test_X)
        return accuracy(pred_y, test_y)
    
    def plot_g_convergence(self, variable = 'g'):
        '''
        Plot:
            - the error of g vs iteration number 
            - the convergence of g (g_{t+1} - g_star)/(g_t - g_star)
        '''
        if self.optim != 'smo':
            print('ERROR: plot not available for this optimisation method')
            return
        
        if variable == 'g':
            iterates = self.g_iterates
        elif variable == 'alpha':
            iterates = self.alpha_iter
        else:
            print(f'Variable not recognised')
        
        
        x_star = iterates[-1] # Use the last value of g obtained as the comparison
        x_min_xstar = np.linalg.norm(iterates-x_star, axis=1)
        nominator = x_min_xstar[1:] # g_{t+1} - g_star
        denominator = x_min_xstar[:-1] # g_t - g_star
        iterates = np.divide(nominator, denominator)
        t = np.arange(1,len(iterates)+1)
        
        # Plot
        plt.figure()
        plt.plot(t, iterates, label='$(g_{t+1}-g^*)/(g_t-g^*)$')
        plt.plot(t, -x_min_xstar[:-1], label='Error $(g_t-g^*)$')
        plt.legend(fontsize = 'large')
        plt.show()
        
    def plot_alpha_convergence(self):
        
        if self.optim != 'barrier':
            print('ERROR: plot not available for this optimisation method')
            return
        
        alpha_star = self.alpha

        iterations = []
        errors = []

        for i in range(len(self.alpha_iter)):
            errors.append(np.linalg.norm(self.alpha_iter[i] - alpha_star))
            iterations.append(i+1)

        plt.figure()
        plt.plot(iterations, errors)
        plt.xlabel('Iterations (k)')
        plt.ylabel(r'$||\alpha_k - \alpha^*||$')
        plt.show()
        
    def plot_duality_gap(self):
        duality_gap = []
        duality_theory = []
        num_iter = 0

        for i in range(len(self.newton_iter)):
            this_dual = self.f_iterates[i] - self.f_iterates[-1]
            this_dual = [this_dual]*self.newton_iter[i]
            duality_gap.extend(this_dual)

            duality_theory.extend([self.duality_gaps_theoretical[i]]*self.newton_iter[i])

            num_iter += self.newton_iter[i]

        iterations = np.arange(1,num_iter+1)

        plt.figure()
        plt.plot(iterations, duality_gap, label=r'$f(\alpha_k) - f(\alpha^*)$')
        plt.plot(iterations, duality_theory, label=r'$m/t$')
        plt.legend(fontsize = 'large')
        plt.xlabel('Cumulative Newton iterations')
        plt.ylabel('Duality gap')
        plt.show()