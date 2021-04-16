import numpy as np
from utils.utils import rbf_kernel, accuracy
from optimisation.smo import fit_smo

class SVM:
    def __init__(self,
                 optim = 'SMO',
                 kernel = 'rbf',
                 C = 1,
                 fit_tol = 1e-3):
        
        # Check that inputs are valid
        if optim not in ['SMO']:
            print("ERROR: optimisation method not supported.")
            return
        if kernel not in ['rbf']:
            print("ERROR: kernel type not supported.")
            return
        
        
        # Init from settings
        self.optim = optim
        self.C = C
        self.kernel_type = kernel
        self.tol = fit_tol
        
        if self.kernel_type == 'rbf':
            self.kernel = rbf_kernel
        
        self.has_been_fit = False # Flag to see if the fit method has been called
        
    def fit(self, train_X, train_y):
        '''
        Fit the SVM on the training data
        '''
        
        
        if self.optim == 'SMO':
            self.alpha, self.b = fit_smo(tol = self.tol,
                                    C = self.C,
                                    kernel=self.kernel,
                                    train_X = train_X,
                                    train_y = train_y,
                                    max_passes=5)
        
        self.train_X = train_X
        self.train_y = train_y    
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
        