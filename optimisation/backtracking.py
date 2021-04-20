import numpy as np

def backtrackig(F, x_k, p, alpha0, rho, c1):
    '''
    Backtracking line search algorithm (see Alg 3.1 in Nocedal & Wright)
    
    Parameters:
    F: class, objective function to be minimised (contains function and gradient)
    x_k: current iterate
    p: descent direction
    alpha0: initial step length
    rho: float in (0,1), backtraking step length reduction factor
    c1: constant in sufficient decrease condition
    
    Returns:
    alpha: step length
    alpha_iterates: step length history
    '''
    
    
    pass