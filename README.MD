- [What is it](#what-is-it)
- [Examples](#examples)
- [Mathematical basis](#mathematical-basis)
  * [European call](#european-call)
  * [Barrier call](#barrier-call)
  * [Lookback call](#lookback-call)

# What is it
This project is made to calculate the following call options:
- European
- Barrier up-and-down (exotic)
- Lookback (exotic)

by 3 methods:
- Monte Carlo simulations
- Exact solution by the Black-Sholes formula
- Numerical solution of the Black-Sholes partial differential equation.

The code is implemented on python. Exact solution for european call and exotic ones rewritten from Shreve (see below), Monte Carlo simulations for exotics uses joint distribution for Wiener process and it's maximum on $[0,T]$, pde solved by Crank-Nicolson scheme. Computational details described below.

This project is made during the MSc program "Financial mathematics and financial technologies" in Sirius University, Sochi, Russia.
# Examples
Examples of using this code coulde be found in example.ipynb notebook.

# Mathematical basis
Calculations mainly based on "Shreve, Stochastic Calculus for Finance II - Continuous Time Models, Springer Finance-v.2, Springer-2004".
For price of call option $V(S, t)$ (where $t$ - time, $S$ - price of underlying asset) we have the following:
## European call
European call is classical option with payoff:
```math
\text{payoff} = \left( S(T) - K \right)_{+}
```
Where $x_{+} = x$ if $x \geq 0$ and 0 if $x < 0$.
### Monte Carlo simulations
```math
V(S(0), T) = \frac{1}{N} \sum_{i = 1}^{N} e^{-r T} \left( S(T) - K \right)_{+}
```
where $N$ - count of iterations, $K$ - strike, $r$ - risk neutrall interest rate, $S(T)$ - generated value of asset $S$ at $t = T$ according to geometric Brownian motion:
```math
S(T) = S_0  e^{ ( r - \frac{\sigma^2}{2} ) T + \sigma  W(T) } 
```
where $W(T) = \sqrt{T} N(0, 1)$ - value of Wiener process at $t = T$, $N(0, 1)$ - standart normal random variable. 
This is just a simulation of call option payoff at $t = T$.

### PDE solution
Black-Sholes PDE for european call option has the form:
```math
V_t + \frac{1}{2} \sigma^2 V_{SS} + r S V_S - r V = 0
```
with initial and border conditions:
```math
\begin{cases} V(S, T) = \left( S(T) - K \right)_{+}, \quad 0 \leq S \leq \infty,
\\ V(0,t) = 0, \quad 0 \leq t \leq T,
\\ V(S, t) \sim S - K e^{-r(T -t)}, \quad S \rightarrow \infty .
\end{cases}
```
This equation could be significantly simplified by variable substitution: 
```math
x = \log{\frac{S}{K}}, \quad \tau = \frac{1}{2} \sigma^2 (T - t), \quad u(x, \tau) = e^{-x} \frac{V(S, t)}{K} .
```
This substitution leads to the equation:
```math
u_{\tau} - \left( u_{xx} + u_{x} \right) - D u_x = 0, \quad D = \frac{2r}{\sigma^2}
```

with initial and border condiditions:
```math
\begin{cases} u(x,0) = \left( 1 - e^{-x} \right)_{+}, \quad x \in (-\infty, \infty),
\\ u(x, \tau) = 0, \quad x \rightarrow - \infty,
\\ u(x, \tau) \sim 1 - e^{-D \tau - x}, \quad x \rightarrow \infty .
\end{cases}
```
The equation above is solved by Crank-Nicolson scheme and tridiagonal matrix algorithm is used for solving system of linear equations problem.

## Barrier call
Barrier call option is considered as exotic option, which value becomes equal to 0, if the value of underlying asset $S$ in any moment $t \leq T$ exceeds the value of pre-known barrier $B$; else it's payoff equals to payoff at of european call option.
### Monte Carlo simulations
```math
V(S(0), T) = \frac{1}{N} \sum_{j = 1}^{N} e^{-r T} \left( S(T) - K \right)_{+} \cdot I_j,
```
where $I_j$ - indicator function, which equals to 0, if asset S exceeded the barrier, or equals to 1 else. 
More precisely this indicator equals to probability to get $Max(\hat{W}_t)$ with specified $\hat{W}_T$ (where maximum is calculated over [0, T] interval]):
```math
P\left( Max( \hat{W}_t) < b \: | \: \hat{W}_T = k \right)  = 1 - e^{2 b \frac{k - b}{T}},
```
where $\hat{W_t} = W_t + a t$ - Wiener process by new measure with $a = \frac{r - \sigma^2 / 2} {\sigma}$ (this process is also Wiener's because of Girsanov's theorem); $b = \frac{1}{\sigma} \log{\frac{B}{S_0}}$ - barrier value for Wiener process, $\hat{W_T} = k$ - value of Wiener process at $t = T$.
The formula above coulde be derived from joint distribution density function for random variables $Max(\hat{W_t}) = m$ and $\hat{W_T} = k$ (see Shreve):
```math
p(m, k) = 2 \frac{(2 m - k)} {T \sqrt{2 \pi T}} e^{a k - \frac{a^2}{2 T} \frac{(2 m - k)^2}{2 T} }
```
by integrating from $-\infty$ to $\infty$ on $k$, and from $-\infty$ to $b$ on $m$.

It should be mentioned, that method described above differs from methods where full trajectories are generated. The latter method is much simplier to implement and do not require knowledge of joint distribution of $W_t$ and $Max(W_t)$. But the crucial problem with this method is that when you generete trajectory and come up close to barrier during time $dt$ you may exceed the barrier and than fall back lower. So you will not establish that you have to reject this trajectory. Therefore by using this simple method you will always get overestimated result. To avoid this you have to increase number of trajectories and number of steps during one trajectory. Computational costs would increse as $O(n_{trajectories}, n_{steps})$. That is very expensive.
The main advantage of method described above is that in every step you only need to generate one random value and calculate probability by simple formula.

### PDE solution
Equation for barrier call is equal to europeans call one:
```math
u_{\tau} - \left( u_{xx} + u_{x} \right) - D u_x = 0, \quad D = \frac{2r}{\sigma^2},
```
but with another initital and border conditions:
```math
\begin{cases} u(x,0) = \left( 1 - e^{-x} \right)_{+}, \quad x \in (-\infty, \infty),
\\ u(x, \tau) = 0, \quad x \rightarrow - \infty,
\\ u(x, \tau) \sim 0, \quad x = \log{\frac{B}{K}}.
\end{cases}
```
This equation also solved by Crank-Nicolson scheme.

## Lookback call
Lookback option is exotic option with floating strike, which payoff depends on $Max(S_t)$ and $S_T$:
```math
\text{payoff} = e^{- r T} \left( Max(S_t) - S_T \right)
```
### Monte Carlo simulations
Calculation is based on generation of $S_T$ by gemetric Brownian motion and generation of $Max(S_t)$. The latter random variable could be generated by using cumulative distribution function (cdf) for $Max(\hat{W}_t)$:
```math
F(x) = P\left( Max( \hat{W}_t) < x \: | \: W_T = \hat{W}_t \right)  = 1 - e^{2 x \frac{K - x}{T}},
```
which is exact probability specified in barrier call section. 
Random variable $Max(S_t) = \xi$ could be derived from uniform distribution by using formula above and fact, that 
```math
F(\xi) = \gamma,
```
which leads to:
```math
\xi = F^{-1}(\gamma)
```
where $\gamma$ is uniformly disturbed on $[0, 1]$. This inverse function is computed explicitly.
So for lookback option price we may write:
```math
V(S(0), T) = \frac{1}{N} \sum_{i = 1}^{N} e^{-r T} \left( Max(S_t) - S_T \right)
```
The arguments given above about advantages of this method of simulations for barrier option also hold for lookback option.

### PDE solution
After operation of dimension reduction (see Shreve) equation for lookback call also follows the pde above. This operation consist of substititution variable $z = \frac{S_t}{Max(S_t)}$. This leads to:
```math
u_{\tau} - \left( u_{zz} + u_{z} \right) - D u_z = 0, \quad D = \frac{2r}{\sigma^2}, \quad z \in [0, 1]
```
with initial and border conditions:
```math
\begin{cases} u(z,0) =  - ( 1 - e^{-z}),
\\ u(0, \tau) = e^{- D \tau} / z, \quad z \rightarrow 0,
\\ u(z, \tau) (e^{z} - 1) = u_{z}(z, \tau), \quad z = 1
\end{cases}
```
After calculations and return to initial variables the value of option price could be computed as: $V(z = 1, t = 0) \cdot S_0$.
