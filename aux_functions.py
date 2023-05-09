import numpy as np
import math
from numba import jit

@jit(nopython = True)
def wiener(t: float) -> float:
    '''Returns value of Wiener process at time = t'''
    return math.sqrt( t ) * np.random.normal(0, 1)

@jit(nopython = True)
def st(t: float, S0: float, r: float, sigma: float) -> float:
    '''Exact solution for geometric Brownian motion for undelying asset s
    
    Parameters.
    1. t - time. Type - float
    2. S0 - value of asset at time = 0. Type - float
    3. r - risk neutral interest rate. Type - float
    4. sigma - volatility. Type - float.

    Output.
    S(t) random variable.
    '''
    return S0 * math.exp( ( r - sigma**2 / 2) * t + sigma * wiener( t ) )

@jit(nopython = True)
def const_tri_diag_mat_solve(params: np.array, mat_b: np.array) -> np.array:
    '''Solution of problem Ax = b by tridiagonal matrix algorithm for matrix with const coefficients.
    
    Parameters.
    1. params - const coefficients of tridiagonal matrix A. Type - numpy array
    2. mat_b - vector of free coefficients b. Type - numpy array.

    Output.
    `x` - vector. Type - numpy array.
    '''
    n = len(mat_b)
    x = np.zeros(n)
    '''
    p1, p2, p3 const parmeters of tridiagonal matrix.
    Matirx correspons to:
    A = sparse.diags([p1, p2, p3], [-1, 0, 1], shape=(n, n))
    '''
    p1, p2, p3 = params[0], params[1], params[2]
    coef_A = np.zeros(n)
    coef_B = np.zeros(n)
    coef_A[1] = - p3 / p2
    coef_B[1] = mat_b[0] / p2
    alpha_gamma = p2 / p1

    '''forward sweep'''
    for i in range(1, n - 1):
        b_gamma = mat_b[i] / p1
        coef_A[i + 1] =  - p3 / ( p1 * coef_A[i] + p2 )
        coef_B[i + 1] = ( b_gamma - coef_B[i] ) / ( coef_A [i] + alpha_gamma) 

    '''backward sweep'''
    i = n - 1
    b_gamma = mat_b[i] / p1
    x[i] = - ( coef_B[i]  - b_gamma ) / ( coef_A[i] + alpha_gamma )
    for j in range(n - 1, 0 , -1):
        x[j - 1] = x[j] * coef_A[j] + coef_B[j]
    return x

@jit(nopython = True)
def tri_diag_mat_solve_arr(p1, p2, p3, mat_b):
    '''Solution of problem Ax = b by tridiagonal matrix algorithm.
    
    Parameters.
    1. p1 - value of diagonal A[i][i - 1]. Type - Numpy Array. First element equals to 0.
    2. p2 - value of diagonal A[i][i]. Type - Numpy Array.
    3. p3 - value of diagonal A[i][i + 1]. Type - Numpy Array. Last element equals to  0.
    4. mat_b - vector of free coefficients. Type - Numpy Array.

    There are actual values in p1 and p3 - (n - 1), but in p2 - n.
    First element equals in p1 - 0, Last element equals in p3 - 0.
    '''
    n = len(mat_b)
    check = len(p1) == len(p3) == len(p2) == n
 
    if check == False:
        raise Exception('wrong dimensions')
    
    x = np.zeros(n)
    coef_A = np.zeros(n)
    coef_B = np.zeros(n)

    coef_A[1] = - p3[0] / p2[0]
    coef_B[1] = mat_b[0] / p2[0]
    
    '''forward sweep'''
    for i in range(1, n - 1):
        b_gamma = mat_b[i] / p1[i]
        alpha_gamma = p2[i] / p1[i]
        coef_A[i + 1] =  - p3[i] / ( p1[i] * coef_A[i] + p2[i] )
        coef_B[i + 1] = ( b_gamma - coef_B[i] ) / ( coef_A [i] + alpha_gamma) 

    '''backward sweep'''
    i = n - 1
    b_gamma = mat_b[i] / p1[i]
    alpha_gamma = p2[i] / p1[i]
    x[i] = - ( coef_B[i]  - b_gamma ) / ( coef_A[i] + alpha_gamma )
    
    for j in range(n - 1, 0 , -1):
        x[j - 1] = x[j] * coef_A[j] + coef_B[j]
    
    return x


def get_result(x_data: np.array, y_data: np.array, val: float) -> float:
    '''Calculate arbitrary function value by grid data.

    # Parameters.
    1. x_data - set of nodes. Type - numpy array
    2. y_data - set of function values at nodes.Type - numpy array
    3. val - value to calculate. Type - float. 
    
    If `val` do not get to the exact node, then between two nearset nodes straight line is drawn and 
    value is calculated in intermediate point.
    '''
    li = np.where(x_data < val)[0][-1]
    ri = np.where(x_data >= val)[0][0]
    if ri == val:
        return y_data[ri]
    else:
        return (y_data[ri] - y_data[li]) / (x_data[ri] - x_data[li]) * (val - x_data[li]) + y_data[li]

def delta_p(tau: float, s: np.array, r: float, sigma: float):
    return 1 / ( sigma * math.sqrt(tau) ) * ( np.log(s) + ( r + 1 / 2 * sigma**2) * tau )

def delta_m(tau: float, s: np.array, r: float, sigma: float):
    return 1 / ( sigma * math.sqrt(tau) ) * ( np.log(s) + ( r - 1 / 2 * sigma**2) * tau )

def cond_prob_M(B, k, T):
    '''Conditional probability function P( Max(W(t)) < B | W(T) = k )'''
    return 1 - math.exp(2 * B * (k - B) / T)