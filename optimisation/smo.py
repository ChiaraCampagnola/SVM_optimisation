import numpy as np
from utils.utils import predict, Q_matrix, calculate_g

class SMO:
    def __init__(self, tol, C, kernel_fun, train_X, train_y, max_passes, speed='fast'):
        self.tol = tol
        self.C = C
        self.kernel_fun = kernel_fun
        self.train_X = train_X
        self.train_y = train_y
        self.max_passes = max_passes
        self.speed = speed
        
    def fit(self):
        if self.speed == 'slow':
            return self.smo_slow()

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

                    # Save old alphas
                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                    # Compute L and H (bound on alpha_j)
                    if self.train_y[i] == self.train_y[j]:
                        L = np.max([0, alpha[i] + alpha[j] - self.C])
                        H = np.min([self.C, alpha[i] + alpha[j]])
                    else:
                        L = np.max([0, alpha[j] - alpha[i]])
                        H = np.max([self.C, self.C + alpha[j] - alpha[i]])
                    
                    if L == H:
                        continue

                    # Compute eta
                    eta = 2*kernel[i,j] - kernel[i,i] - kernel[j,j]

                    if eta >= 0:
                        continue
                    
                    # Compute new alpha_j and clip
                    alpha_j = alpha_j_old - (self.train_y[j] * (E_i - E_j))/eta
                    if alpha_j > H:
                        alpha_j = H
                    elif alpha_j < L:
                        alpha_j = L
                    
                    alpha[j] = alpha_j

                    # If alpha_j has not changed "significantly", end iteration
                    if np.abs(alpha_j - alpha_j_old) < 1e-5:
                        continue

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

def smo_fast(tol, C, kernel_fun, train_X, train_y, max_passes):
    num_training_examples = train_y.shape[0]

    # Initialise
    alpha = np.zeros(num_training_examples)
    b = 0
    passes = 0
    
    kernel = kernel_fun(train_X, train_X)
    
    g_iterates = []
    
    while passes < max_passes:
        changed_alpha = False
        for i in range(num_training_examples): # Iterate through alphas

            ########## Check if alpha_i violates KKT conditions ########
            pred_y_i = predict(alpha, kernel[:,i], b, train_y)
            E_i = pred_y_i - train_y[i] # Error

            if (train_y[i]*E_i < -tol and alpha[i] < C) or (train_y[i]*E_i > -tol and alpha[i] > 0): # KKT conditions are broken

                # Pick alpha_j != alpha_i at random
                j = np.random.randint(0, num_training_examples)
                while j == i:
                    j = np.random.randint(0, num_training_examples)

                # Calculate error 2
                pred_y_j = predict(alpha, kernel[:,j], b, train_y)
                E_j = pred_y_j - train_y[j]

                # Save old alphas
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                # Compute L and H (bound on alpha_j)
                if train_y[i] == train_y[j]:
                    L = np.max([0, alpha[i] + alpha[j] - C])
                    H = np.min([C, alpha[i] + alpha[j]])
                else:
                    L = np.max([0, alpha[j] - alpha[i]])
                    H = np.max([C, C + alpha[j] - alpha[i]])
                
                if L == H:
                    continue

                # Compute eta
                eta = 2*kernel[i,j] - kernel[i,i] - kernel[j,j]

                if eta >= 0:
                    continue
                
                # Compute new alpha_j and clip
                alpha_j = alpha_j_old - (train_y[j] * (E_i - E_j))/eta
                if alpha_j > H:
                    alpha_j = H
                elif alpha_j < L:
                    alpha_j = L
                
                alpha[j] = alpha_j

                # If alpha_j has not changed "significantly", end iteration
                if np.abs(alpha_j - alpha_j_old) < 1e-5:
                    continue

                # Compute new alpha_i
                alpha[i] = alpha_i_old + train_y[i] * train_y[j] * (alpha_j_old - alpha_j)

                # Compute new threshold
                b1 = b - E_i - train_y[i] * (alpha[i] - alpha_i_old) * kernel[i,i] - train_y[j] * (alpha[j] - alpha_j_old) * kernel[i,j]
                b2 = b - E_j - train_y[i] * (alpha[i] - alpha_i_old) * kernel[i,j] - train_y[j] * (alpha[j] - alpha_j_old) * kernel[j,j]

                if alpha[i] > 0 and alpha[i] < C:
                    b = b1
                elif alpha[j] > 0 and alpha[j] < C:
                    b - b2
                else:
                    b = (b1+b2)/2
                
                changed_alpha = True
                
                g_iterates.append(calculate_g(alpha, Q_matrix(train_X, train_y, kernel_fun)))
                
        if not changed_alpha:
            passes += 1
            
    # Find support vectors so we keep only elements corresponding to suport vectors
    SV = alpha > 0
    
    result = {'train_X': train_X[SV,:],
              'train_y': train_y[SV],
              'alpha': alpha[SV],
              'b': b,
              'g': g_iterates}
    
    return result

