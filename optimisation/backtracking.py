import numpy as np

def backtracking(F, x_k, p, alpha0, rho=0.1, c1=1e-4):
    '''
    Backtracking line search algorithm (see Alg 3.1 in Nocedal & Wright)
    
    Parameters:
    F: class, objective function to be minimised (contains function and gradient)
    x_k: current iterate
    p: descent direction (vector)
    alpha0: initial step length
    rho: float in (0,1), backtraking step length reduction factor
    c1: constant in sufficient decrease condition
    
    Returns:
    alpha: step length
    alpha_iterates: step length history
    '''
    # Init
    alpha_iterates = [alpha0]
    alpha = alpha0
    
    while F.f(x_k + alpha*p) > F.f(x_k) + c1 * alpha * np.dot(F.df(x_k), p): # Stopping condition
        alpha = rho * alpha
        alpha_iterates.append(alpha)
    
    return alpha, alpha_iterates