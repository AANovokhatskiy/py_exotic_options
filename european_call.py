import numpy as np
import math
from scipy.stats import norm
from aux_functions import const_tri_diag_mat_solve, get_result, st

class european_call_option:
    '''
    European call option

    # Parameters.
        1. T - expiration time. Type - float
        2. t - time. Type - float
        3. S0 - value of asset at time = 0. Type - float
        4. K - strike. Type - float
        5. r - risk neutral interest rate. Type - float
        6. sigma - volatility. Type - float.

    # List of available methods
    1. price_exact - exact solution of Black-Sholes equation
    2. price_monte_carlo - Monte-Carlo simulation of option price
    3. price_pde - numerical solution of Black-Sholes PDE
    4. get_pde_price - calculation of option price at point S0.
    '''
    def __init__(self, T: float, t: float, S0: float, K: float, r: float, sigma: float) -> None:
        self.verify_init_data(T, t, S0, K, r, sigma)
        self.__T = T
        self.__t = t 
        self.__S0 = S0
        self.__K = K
        self.__r = r
        self.__sigma = sigma

        #self.__mc_v = np.nan
        #self.__exact_v = np.nan

        self.__pde_calc_flg = 0
        self.__pde_t = np.nan
        self.__pde_s = np.nan
        self.__pde_v = np.nan

    @classmethod
    def verify_init_data(cls, T, t, S0, K, r, sigma):
        params = [T, S0, K, r, sigma]
        names = ['T', 'S0', 'K', 'r', 'sigma']
        n = len(params)
        for i in range(0, n):
            param_type = type(params[i])
            if not (param_type == int or param_type == float):
                raise TypeError(f"{names[i]} should be a number, got {param_type.__name__}")
            if params[i] <= 0:
                raise TypeError(f"{names[i]} should be a positive number, got {params[i]}")
            
        '''handle t'''
        param_type = type(t)
        if not (param_type == int or param_type == float):
            raise TypeError(f"{names[i]} should be a number, got {param_type.__name__}")
        if t < 0:
            raise TypeError(f"{names[i]} should be a positive number, got {t}")        
        if t > T: 
            raise TypeError(f"t is out of [0, T] interval. got [0, {T}] and t = {t}")

    @property
    def T(self):
        return self.__T

    @property
    def t(self):
        return self.__t
    
    @property
    def S0(self):
        return self.__S0
    
    @property
    def K(self):
        return self.__K
        
    @property
    def r(self):
        return self.__r
    
    @property
    def sigma(self):
        return self.__sigma

    @property
    def pde_t(self):
        return self.__pde_t
    
    @pde_t.setter
    def pde_t(self, arr):
        self.__pde_t = arr

    @property
    def pde_s(self):
        return self.__pde_s
    
    @pde_s.setter
    def pde_s(self, arr):
        self.__pde_s = arr

    @property
    def pde_v(self):
        return self.__pde_v

    @pde_v.setter
    def pde_v(self, arr):
        self.__pde_v = arr

    @property
    def pde_calc_flg(self):
        return self.__pde_calc_flg

    @pde_calc_flg.setter
    def pde_calc_flg(self, val):
        self.__pde_calc_flg = val

    #@jit(nopython = True)
    def price_monte_carlo(self, n_iters: int):
        '''Monte Carlo simulaton of european call option price.
           t parameter is not considered in this function.

        Parameters.
        n_iters - count of monte-carlo iterations. Type - Int.

        Output.
        Average price.
        '''
        mx = 0
        r = self.r
        T = self.T
        S0 = self.S0
        K = self.K
        sigma = self.sigma
        for i in range(0, n_iters):
            b = math.exp(-r * T) * max( st(T, S0, r, sigma) - K, 0)
            mx += b
        return mx / n_iters    

    def price_exact(self):
        '''Exact solution of Black-Sholes PDE for european call option price.

        Output.
        V(S0, T) price of call option at time = `T` and initial underlying price = `S0`.
        '''

        tau = self.T - self.t
        dp = 1 / ( self.sigma * math.sqrt( tau ) ) * ( math.log( self.S0 / self.K ) + ( self.r + self.sigma**2 / 2) * ( tau ) )
        dm = dp - self.sigma * math.sqrt( tau )
        return self.S0 * norm.cdf(dp) - self.K * math.exp( - self.r * tau ) * norm.cdf(dm)
            
    #@jit(nopython = True)
    def price_pde(self, n_t: int, n_s: int) -> np.array:
        '''
        Solution of u_t = u_xx + u_x * ( 1 + D ). D = 2r / s^2. Crank-Nicolson scheme.
        PDE is considered in tau = sigma**2 / 2 * (T - t) and x = log (S/K) variables.
        Transition to initial variables is made at the end of evaluations.
        Initial parameters (`S0`, `t`) of call_option class is not considered in this function.

        # Parameters.
        1. n_t - number of `t` grid steps. Type - Int
        2. n_s - number of `S` grid steps. Type - Int.

        # Output.
        Returns set of numpy arrays: 
        1. t - array with length of n_t (corresponds to region [0, T])
        2. s - array with lengh of n_s (corresponds to region [K / 3, 3 * K])
        3. v - matrix of call option price at t_i, x_j
        '''

        '''Auxilary parameters'''
        T, K, r, sigma = self.T, self.K, self.r, self.sigma
        n_x = n_s
        region = [[0, T], [K / 3, 3 * K]]
        right_t, left_t = sigma**2 / 2 * (T - region[0][0]), sigma**2 / 2 * (T - region[0][1])
        left_x, right_x = math.log(region[1][0] / K), math.log(region[1][1] / K)
        tau = abs(right_t - left_t) / n_t
        h = abs(right_x - left_x) / n_x
        u = np.zeros((n_t, n_x))
        D =  2 * r / sigma**2
        
        '''Initial and border conditions.'''

        '''x = -inf'''
        for i in range(0, n_t): 
            u[i][0] = 0
        '''x = inf'''
        for i in range(0, n_t):
            u[i][n_x - 1] = 1 - math.exp( - D * (left_t + i * tau ) - right_x )
        '''tau = 0'''
        for i in range(0, n_x):
            u[0][i] = max(0 , 1 - math.exp(- ( left_x + i * h ) ) )

        '''Set matrix for solving system of linear equations'''
        size = n_x - 2
        p1, p2, p3 = - tau, 2 * h**2 + 2 * tau + tau * h * (1 + D), - tau - tau * h * (1 + D)
        A_mat_params = np.array([p1, p2, p3])

        p4 = 2 * h**2 - 2 * tau - tau * h * (1 + D)
        '''Finite difference scheme'''
        for k in range(0, n_t - 1):
            '''Vector of free coefficients b'''
            b = np.zeros(size)
            b[0] = - p1 * u[k + 1][0] - p1 * u[k][0] + p4 * u[k][1] - p3 * u[k][2]
            b[1:size - 1] = - p1 * u[k][1:size - 1] + p4 * u[k][2:size] - p3 * u[k][3:size + 1]
            b[size - 1] = - p3 * u[k + 1][n_x - 1] - p1 * u[k][n_x - 3] + p4 * u[k][n_x - 2] - p3 * u[k][n_x - 1]
            '''Solving system Ax = b by tridiagonal matrix algorithm'''
            res = const_tri_diag_mat_solve(A_mat_params, b)
            u[k + 1][1:n_x - 1] = res
            
        '''Transition from function u(x, tau) to V(S,t)'''
        x_data = np.exp(np.linspace(left_x, right_x, n_x)) * K
        for k in range(0, n_t):
            u[k] = u[k] * x_data
        
        '''Transition from coordinates tau to t'''
        T1 = sigma**2 * T / 2
        t = np.linspace(0, T1, n_t)
        t = T - 2 * t / sigma**2

        '''Transition from coordinates x to S'''
        s = np.linspace(math.log( region[1][0] / K ), math.log( region[1][1] / K), n_x)
        s = np.exp(s) * K

        self.pde_t = t
        self.pde_s = s
        self.pde_v = u
        
        self.pde_calc_flg = 1

        return (t, s, u)

    def get_pde_result(self, S0: float = None):
        '''Returns pde call option price v(S0, t)'''
        if self.pde_calc_flg == 0:
            raise ValueError(f'Nothing to return. Method price_pde should be called first.')
        if S0 == None:
            S0 = self.S0
        return get_result(self.pde_s, self.pde_v[-1], S0)