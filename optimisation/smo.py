import numpy as np
from utils.utils import predict, Q_matrix, calculate_g

class SMO:
    def __init__(self, tol, C, kernel_fun, train_X, train_y, max_passes, speed='fast', calc_g_iterates=False):
        self.tol = tol
        self.C = C
        self.kernel_fun = kernel_fun
        self.train_X = train_X
        self.train_y = train_y
        self.max_passes = max_passes
        self.speed = speed
        self.calc_g_iterates = calc_g_iterates
        
    def fit(self):
        if self.speed == 'slow':
            return self.smo_slow()
        if self.speed == 'fast':
            return self.smo_fast()
        
    def optimise_alphas(self, alpha, b, kernel, i, j, E_i, E_j):
        changed_alpha = False
        # Save old alphas
        alpha_i_old, alpha_j_old = alpha[i], alpha[j]

        # Compute L and H (bound on alpha_j)
        if self.train_y[i] == self.train_y[j]:
            L = np.max([0, alpha[i] + alpha[j] - self.C])
            H = np.min([self.C, alpha[i] + alpha[j]])
        else:
            L = np.max([0, alpha[j] - alpha[i]])
            H = np.min([self.C, self.C + alpha[j] - alpha[i]])
        
        #print(f'H: {H}, L: {L}')
        
        if L != H:
            # Compute eta
            eta = 2*kernel[i,j] - kernel[i,i] - kernel[j,j]

            if eta < 0:
                # Compute new alpha_j and clip
                alpha_j = alpha_j_old - (self.train_y[j] * (E_i - E_j))/eta
                if alpha_j > H:
                    alpha_j = H
                elif alpha_j < L:
                    alpha_j = L
                
                alpha[j] = alpha_j

                # Only continue if alpha_j has changed "significantly"
                if np.abs(alpha_j - alpha_j_old) > 1e-5:
                    # Compute new alpha_i
                    alpha[i] = alpha_i_old + self.train_y[i] * self.train_y[j] * (alpha_j_old - alpha_j)

                    # Compute new threshold
                    b1 = b - E_i - self.train_y[i] * (alpha[i] - alpha_i_old) * kernel[i,i] - self.train_y[j] * (alpha[j] - alpha_j_old) * kernel[i,j]
                    b2 = b - E_j - self.train_y[i] * (alpha[i] - alpha_i_old) * kernel[i,j] - self.train_y[j] * (alpha[j] - alpha_j_old) * kernel[j,j]

                    if alpha[i] > 0 and alpha[i] < self.C:
                        b = b1
                    elif alpha[j] > 0 and alpha[j] < self.C:
                        b - b2
                    else:
                        b = (b1+b2)/2
                    
                    changed_alpha = True
        return alpha, b, changed_alpha

    def smo_slow(self):
        num_training_examples = self.train_y.shape[0]

        # Initialise
        alpha = np.zeros(num_training_examples)
        b = 0
        passes = 0
        
        kernel = self.kernel_fun(self.train_X, self.train_X)
        
        g_iterates = []
        
        while passes < self.max_passes:
            changed_alpha = False
            for i in range(num_training_examples): # Iterate through alphas

                ########## Check if alpha_i violates KKT conditions ########
                pred_y_i = predict(alpha, kernel[:,i], b, self.train_y)
                E_i = pred_y_i - self.train_y[i] # Error

                if (self.train_y[i]*E_i < -self.tol and alpha[i] < self.C) or (self.train_y[i]*E_i > -self.tol and alpha[i] > 0): # KKT conditions are broken

                    # Pick alpha_j != alpha_i at random
                    j = np.random.randint(0, num_training_examples)
                    while j == i:
                        j = np.random.randint(0, num_training_examples)

                    # Calculate error 2
                    pred_y_j = predict(alpha, kernel[:,j], b, self.train_y)
                    E_j = pred_y_j - self.train_y[j]
                    
                    alpha, b, changed_alpha = self.optimise_alphas(alpha, b, kernel, i, j, E_i, E_j)
                    
                    if self.calc_g_iterates:
                        g_iterates.append(calculate_g(alpha, Q_matrix(self.train_X, self.train_y, self.kernel_fun)))
                    
            if not changed_alpha:
                passes += 1
                
        # Find support vectors so we keep only elements corresponding to suport vectors
        SV = alpha > 0
        
        result = {'train_X': self.train_X[SV,:],
                'train_y': self.train_y[SV],
                'alpha': alpha[SV],
                'b': b,
                'g': g_iterates}
        
        return result

    def candidate_alphas(self, alpha, kernel, b):
        # Return two sets:
        #   - alphas breaking KKT
        #   - non-boundary alphas breaking  KKT
        
        self.E = {}
        
        candidates = set()
        non_bound_candidates = set()
        
        num_training_examples = self.train_y.shape[0]

        for i in range(num_training_examples): # Iterate through alphas

            if self.breaks_KKT(alpha, kernel, b, i):
                candidates.add(i)
                
                # Check it's non-bound
                if alpha[i] != 0 and alpha[i] != self.C:
                    non_bound_candidates.add(i)
                
        return candidates, non_bound_candidates
    
    # def find_alpha_2_old(self, i, alpha, kernel, b, non_bound_alphas):
    #     pred_y_i = predict(alpha, kernel[:,i], b, self.train_y)
    #     E_i = pred_y_i - self.train_y[i] # Error
        
    #     found = False
        
    #     if E_i > 0:
    #         min_E = 100000000
    #         best_j = 0
    #         for j in non_bound_alphas:
    #             if self.E[j] < min_E and j!=i:
    #                 min_E = self.E[j]
    #                 best_j = j
    #                 found = True
    #     else:
    #         max_E = -1000000
    #         best_j = 0
    #         for j in non_bound_alphas:
    #             if self.E[j] > max_E and j!=i:
    #                 max_E = self.E[j]
    #                 best_j = j
    #                 found = True
    #     return best_j
    
    def breaks_KKT(self, i):
        r = self.E[i] * self.train_y[i]
        
        if (r < -self.tol and self.alpha[i] < self.C) or (r > self.tol and alpha[i] > 0):
            return True
        return False
    
    def calculate_Es(self):
        y_pred = np.sign(np.multiply(self.alpha, self.train_y) @ self.train_kernel + self.b)
        Es = y_pred - self.train_y
        for i, E in enumerate(Es):
            self.E[i] = E
            if E > self.maxE[0]:
                self.maxE[0] = E
                self.maxE[1] = i
            elif E < self.minE[0]:
                self.minE[0] = E
                self.minE[1] = i
             
    def find_alpha_2(self, j):
        # First heuristic
        self.calculate_Es()
        if self.E[j] > 0:
            i = self.minE[1]
        else:
            i = self.maxE[1]
        if self.take_step(j, i):
            return 1
        
        # Second heuristic
        rand_idx = np.random.permutation(len(self.train_y))
        for i in rand_idx:
            if self.alpha[i] > 0 and self.alpha[i] < self.C and i != j:
                if self.take_step(j,i):
                    return 1
                
        # Third heuristic
        rand_idx = np.random.permutation(len(self.train_y))
        for i in rand_idx:
            if i != j:
                if self.take_step(j,i):
                    return 1
                
        return 0 
        
        
        
    def take_step(self, j, i):
        if i==j:
            return 0 # No step taken
        
        if self.train_y[i] == self.train_y[j]:
            L = np.max([0, self.alpha[j] + self.alpha[i] - self.C])
            H = np.min([self.C, self.alpha[j] + self.alpha[i]])
        else:
            L = np.max([0, self.alpha[j] - self.alpha[i]])
            H = np.min([self.C, self.C + self.alpha[j] - self.alpha[i]])
            
        if L == H:
            return 0
        
        eta = 2*self.train_kernel[i,j] - self.train_kernel[i,i] - self.train_kernel[j,j]
        
        if eta >= 0: # TO CHECK
            return 0
        
        alpha_i_old = self.alpha[i]
        alpha_j_old = self.alpha[j]
        
        self.alpha[j] = alpha_j_old - self.train_y[j]*(self.E[i] - self.E[j])/eta
        
        if self.alpha[j]  > H:
            self.alpha[j]  = H
        elif self.alpha[j]  < L:
            self.alpha[j]  = L
        
        self.alpha[i] = alpha_i_old + self.train_y[i]*self.train_y[j]*(alpha_j_old - self.alpha[j])
        
        # Compute new threshold
        b1 = self.b - self.E[i] - self.train_y[i] * (self.alpha[i] - alpha_i_old) * self.train_kernel[i,i] - self.train_y[j] * (self.alpha[j] - alpha_j_old) * self.train_kernel[i,j]
        b2 = self.b - self.E[j] - self.train_y[i] * (self.alpha[i] - alpha_i_old) * self.train_kernel[i,j] - self.train_y[j] * (self.alpha[j] - alpha_j_old) * self.train_kernel[j,j]

        if self.alpha[i] > 0 and self.alpha[i] < self.C:
            self.b = b1
        elif self.alpha[j] > 0 and self.alpha[j] < self.C:
            self.b = b2
        else:
            self.b = (b1+b2)/2
        
        if self.calc_g_iterates:
            self.g_iterates.append(calculate_g(self.alpha, Q_matrix(self.train_X, self.train_y, self.kernel_fun, self.train_kernel)))
        return 1
        
         
    def examine_example(self, j):
        self.E[j] = predict(self.alpha, self.train_kernel[:,j], self.b, self.train_y) - self.train_y[j]
        if self.breaks_KKT(j):
            i = self.find_alpha_2(j)
            return self.take_step(j,i)
        return 0

    def smo_fast(self):
        num_training_examples = self.train_y.shape[0]

        # Initialise
        self.alpha = np.zeros(num_training_examples)
        self.b = 0
        self.E = {}
        self.maxE = [-10000000, 0]
        self.minE = [10000000, 0]
        
        self.train_kernel = self.kernel_fun(self.train_X, self.train_X)
        
        self.g_iterates = []
        
        numChanged = 0
        examineAll = True
        
        passes = 0
        
        while (passes < self.max_passes):
            numChanged = 0
            
            # Go through all examples
            if examineAll:
                for j in range(num_training_examples):
                    numChanged += self.examine_example(j)
            
            # Go through non bound examples      
            else:
                for j in range(num_training_examples):
                    if self.alpha[j] != 0 and self.alpha[j] != self.C:
                        numChanged += self.examine_example(j)
            if numChanged == 0:
                passes += 1
            if examineAll is True: # If we were going through big loop, switch to small
                examineAll = False
            elif numChanged == 0: # If we were going through small loop and we haven't changed any, switch to large
                examineAll = True
            
        # Find support vectors so we keep only elements corresponding to suport vectors
        #print(self.alpha)
        SV = self.alpha > 0
        
        result = {'train_X': self.train_X[SV,:],
                'train_y': self.train_y[SV],
                'alpha': self.alpha[SV],
                'b': self.b,
                'g': self.g_iterates}
        
        return result