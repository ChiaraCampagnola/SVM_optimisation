import numpy as np

def newton_method(F, x0, A, b, line_search, max_iter=200, tol=1e-4, alpha0 = 1):
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
        
        # Find Newton step
        delta_x, w = newton_step(F, A, b, x_k)
        
        # Check stopping criterion
        lambda_k_squared = np.transpose(delta_x) @ F.d2f(x_k) @ delta_x
        
        if lambda_k_squared/2 <= tol:
            break
        
        # Run line search to find step size
        alpha, _ = line_search(F, x_k, delta_x, alpha0)
        
        # Update x
        x_k = x_k + alpha * delta_x
        
    return x_k, nIter

def newton_step(F, A, b, x):
    '''
    Find the newton step using section 10.2.1 in Boyd:
    
    | F.d2f(x)   A^T | | delta_x | = | -F.df(x) |
    |    A       0   | |    w    |   |     b    |
    
            M               v              h
     
    '''
    d2f = F.d2f(x)
    r = d2f.shape[0]
    c = A.shape[0]
    
    m1 = np.vstack((d2f, A))
    m2 = np.vstack((np.transpose(A), np.zeros((r,c))))
    M = np.hstack((m1,m2))
    
    h = np.hstack((-F.df(x), b))
    
    try:
        res = np.linalg.solve(M,h)
    except:
        res = np.linalg.pinv(M) @ h
        
    delta_x = res[:len(x)]
    w = res[len(x):]
    
    return delta_x, w