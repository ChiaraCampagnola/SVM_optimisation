import numpy as np
from utils.utils import Q_matrix, calculate_g

class SMO:
    def __init__(self, tol, C, kernel_fun, train_X, train_y, max_passes, calc_g_iterates=False):
        # Parameters
        self.tol = tol
        self.C = C
        self.kernel_fun = kernel_fun
        self.train_X = train_X
        self.train_y = train_y
        self.max_passes = max_passes
        self.calc_g_iterates = calc_g_iterates
        
        # Initialisations
        self.E = {}
        self.kernel = self.kernel_fun(self.train_X, self.train_X)
        self.alpha = np.zeros(self.train_y.shape[0])
        self.b = 0
        self.g_iterates = []
        
        
    def fit(self):
        num_training_examples = self.train_y.shape[0]

        # Initialise
        passes = 0
        
        while passes < self.max_passes:
            changed_alpha = False
            for i in range(num_training_examples): # Iterate through alphas
                pred_y_i = self.predict(i)
                self.E[i] = pred_y_i - self.train_y[i] # Error

                if self.breaks_KKT(i): # KKT conditions are broken

                    # Pick alpha_j != alpha_i at random
                    j = np.random.randint(0, num_training_examples)
                    while j == i:
                        j = np.random.randint(0, num_training_examples)

                    # Calculate error 2
                    pred_y_j = self.predict(j)
                    self.E[j] = pred_y_j - self.train_y[j]
                    
                    changed_alpha = self.optimise_alphas(i, j)
                    
                    if self.calc_g_iterates:
                        self.g_iterates.append(calculate_g(self.alpha, Q_matrix(self.train_X, self.train_y, self.kernel_fun)))
                    
            if not changed_alpha:
                passes += 1
                
        # Find support vectors so we keep only elements corresponding to suport vectors
        SV = self.alpha > 0
        
        result = {'train_X': self.train_X[SV,:],
                'train_y': self.train_y[SV],
                'alpha': self.alpha[SV],
                'b': self.b,
                'g': self.g_iterates}
        
        return result
        
        
    def optimise_alphas(self, i, j):
        changed_alpha = False
        # Save old alphas
        alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

        # Compute L and H (bound on alpha_j)
        if self.train_y[i] == self.train_y[j]:
            L = np.max([0, self.alpha[i] + self.alpha[j] - self.C])
            H = np.min([self.C, self.alpha[i] + self.alpha[j]])
        else:
            L = np.max([0, self.alpha[j] - self.alpha[i]])
            H = np.min([self.C, self.C + self.alpha[j] - self.alpha[i]])
        
        if L != H:
            # Compute eta
            eta = 2*self.kernel[i,j] - self.kernel[i,i] - self.kernel[j,j]

            if eta < 0:
                # Compute new alpha_j and clip
                alpha_j = alpha_j_old - (self.train_y[j] * (self.E[i] - self.E[j]))/eta
                if alpha_j > H:
                    alpha_j = H
                elif alpha_j < L:
                    alpha_j = L
                
                self.alpha[j] = alpha_j

                # Only continue if alpha_j has changed "significantly"
                if np.abs(alpha_j - alpha_j_old) > 1e-5:
                    # Compute new alpha_i
                    self.alpha[i] = alpha_i_old + self.train_y[i] * self.train_y[j] * (alpha_j_old - alpha_j)

                    # Compute new threshold
                    b1 = self.b - self.E[i] - self.train_y[i] * (self.alpha[i] - alpha_i_old) * self.kernel[i,i] - self.train_y[j] * (self.alpha[j] - alpha_j_old) * self.kernel[i,j]
                    b2 = self.b - self.E[j] - self.train_y[i] * (self.alpha[i] - alpha_i_old) * self.kernel[i,j] - self.train_y[j] * (self.alpha[j] - alpha_j_old) * self.kernel[j,j]

                    if self.alpha[i] > 0 and self.alpha[i] < self.C:
                        self.b = b1
                    elif self.alpha[j] > 0 and self.alpha[j] < self.C:
                        self.b - b2
                    else:
                        self.b = (b1+b2)/2
                    
                    changed_alpha = True
        return changed_alpha
    
    def breaks_KKT(self, i):
        r = self.E[i] * self.train_y[i]
        
        if (r < -self.tol and self.alpha[i] < self.C) or (r > self.tol and alpha[i] > 0):
            return True
        return False
    
    def predict(self, i = None):
        
        # If an index is given, predict that example
        if i is not None:
            return np.sign(np.multiply(self.alpha, self.train_y) @ self.kernel[:,i] + self.b)
        
        # ... otherwise predict all
        return np.sign(np.multiply(self.alpha, self.train_y) @ self.kernel + self.b)