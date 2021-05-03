import numpy as np

def newton_method(F, x0, A, C, line_search, max_iter=10, tol=1e-4, step_length0 = 0.1):
    '''
    Newton's method with equality constraint
    Algorithm 10.1 in Boyd
    
    Arguments:
    F: objective function (class)
    x0: starting point
    A: matrix from the equality constraint
    line_search: function to perform line search
    max_iter: int (for safety stopping condition)
    tol: for stopping condition
    
    Returns:
    x_min: solution of min problem
    f_min: value of f at x_min
    x_iterates: list of x_k iterates
    '''
    # Init
    x_k = x0
    nIter = 0
    x_iterates = [x0]
    
    while nIter < max_iter:
        nIter += 1
        
        # Find Newton step direction
        step_direction, _ = newton_step(F, A, x_k)
                
        # Check stopping criterion
        lambda_k_squared = np.transpose(step_direction) @ F.d2f(x_k) @ step_direction
        
        if lambda_k_squared/2 <= tol:
            break
        
        # Find initial step length (need to make sure it doesn't break F, ie: no alphas are = 0 or >= C)
        step_length0 = initial_step_length(x_k, C, step_direction)
        
        # Run line search to find step size
        step_length, iterates = line_search(F, x_k, step_direction, step_length0)
        
        # Update x
        x_k = x_k + step_length * step_direction
    
    #print(f'niter: {nIter}')
    return x_k, nIter

def newton_step(F, A, x):
    '''
    Find the newton step using section 10.2.1 in Boyd:
    
    | F.d2f(x)   A^T | | delta_x | = | -F.df(x) |
    |    A       0   | |    w    |   |     0    |
    
            M               v              h
     
    '''
    
    d2f = F.d2f(x)
    df = F.df(x)
    r = d2f.shape[0]
    c = A.shape[0]
    
    m1 = np.vstack((d2f, A))
    m2 = np.vstack((A.reshape(-1,1), 0))

    M = np.hstack((m1,m2))
 
    h = np.vstack((-df.reshape(-1,1), 0))
    
    try:
        res = np.linalg.solve(M,h)
    except:
        res = np.linalg.pinv(M) @ h
        
    delta_x = res[:len(x)].squeeze()
    w = res[len(x):].squeeze()
    
    return delta_x, w

def initial_step_length(x_k, C, step_direction):
    step = 0.5
    while np.any(x_k + step*step_direction <= 0) or np.any(x_k + step*step_direction >= C):
        step = 0.8*step
        
    return step