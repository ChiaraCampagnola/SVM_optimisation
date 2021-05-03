import numpy as np

def backtracking(F, x_k, p, step_length0, rho=0.5, c1=1e-4):
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
    step: step length
    step_iterates: step length history
    '''
    # Init
    step_iterates = [step_length0]
    step = step_length0
    
    while F.f(x_k + step*p) > F.f(x_k) + c1 * step * np.dot(np.transpose(F.df(x_k)), p): # Stopping condition
        step = rho * step
        step_iterates.append(step)
    
    return step, step_iterates