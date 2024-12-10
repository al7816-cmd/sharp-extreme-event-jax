# Source Code for "Scalability of the second-order reliability method for stochastic differential equations with multiplicative noise"

### Problem formulation

The code in this repository can be used to determine the tail probability
$\mathbb{P} \left[ f(X(T), Y(T)) \geq z \right]$
for the two-dimensional stochastic differential equation (a predator-prey model)

$$
\begin{cases}
    \mathrm{d} X = (-\beta XY + \alpha X + \delta) \mathrm{d} t + \sqrt{\varepsilon}\sqrt{\beta X Y + \alpha X + \delta} \mathrm{d} W_x\\
    \mathrm{d} Y = (+\beta XY - \gamma Y + \delta) \mathrm{d} t + \sqrt{\varepsilon}\sqrt{\beta X Y + \gamma Y + \delta} \mathrm{d} W_y
  \end{cases}
$$

with deterministic initial condition $(X(0),Y(0)) = (x_0,y_0)$ at the unique stable fixed point of the deterministic system for the chosen rates $\alpha, \beta, \gamma, \delta$, and observable
$f(x, y) = x$, meaning we are interested in large prey concentrations. The code contains functions to obtain sampling estimates,
as well as to compute an asymptotic estimate of the tail probability using the instanton and the leading-order prefactor. The calculations for
the latter are implemented using JAX and rely on automatic differentiation.

### Standard Parameters

These are the default parameters that were also used in the first section of the paper:
* $T = 10$
* $\varepsilon = 0.01$
* $z = 0.5$
* $n_t = 1000$
* $\alpha = 1$
* $\beta  = 5$
* $\gamma = 1$
* $\delta = 0.1$

### Prerequisites

The python3 script requires numpy, scipy, matplotlib and JAX.
They were tested under Python 3.10.12, numpy 1.26.2, scipy 1.11.4 and matplotlib 3.8.2 and JAX 0.4.21 (compiled on CPU).
