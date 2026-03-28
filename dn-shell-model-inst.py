import sys
import copy
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.stats import norm
import jax
import jax.numpy as jnp
import scipy.optimize as spo
import tqdm
jax.config.update("jax_traceback_filtering", "off")

######################################################################
# set up parameters for model

T = 1
dt = 1e-4
nt = int(T / dt)
t = jnp.linspace(0., T, nt + 1)

# DN shell model parameters
dim = 17    # number of discretized concentric shells n = 0 ... dim - 1
c1 = 1.0    # constant parameter for getG
c2 = 0.0    # constant parameter for getG
nu = 1e-2   # viscosity
k = 2 ** jnp.linspace(0, dim-1, dim) # wavenumbers
sigma = 1.  # noise strength for direct sampling
chi_sqrt = sigma * k**(-3.)  # forcing cov sqrt

# initial condition: start at fixed point
x0 = 1. / (5. * jnp.sqrt(2.))
y0 = x0 + 0.2
init_u = jnp.zeros(dim)

targetObs = 0.5

print('Running Instanton computation for DN shell model')
print(f'Parameters: T={T}, dt={dt}, sigma^2={sigma ** 2.}, dim={dim}')


plots = True # option for producing plots of instanton and operator spectra
######################################################################
# define DN shell model functions

jgetIF = lambda u, dt: u * jnp.exp(-nu * k[:, None]**2 * dt)  # integrating factor for linear terms in b times x
jgetIF_single = lambda u, dt: u * jnp.exp(-nu * k**2 * dt)

jgetEnergy = lambda u: 0.5 * u**2
jgetEnergyDissipation = lambda u : jnp.sum(nu * k[:,None]**2 * u**2, axis = 0)


# to be used in MC computation
def jgetG(u):
    
    # :param u: (dim, nPaths)
    # :return: (dim, nPaths)
    
    Gu = jnp.zeros_like(u)

    Gu = Gu.at[1:].add(c1 * k[1:, None] * u[:-1] * u[:-1])
    Gu = Gu.at[:-1].add(-c1 * k[1:, None] * u[:-1] * u[1:])
    Gu = Gu.at[1:].add(c2 * k[1:, None] * u[:-1] * u[1:])
    Gu = Gu.at[:-1].add(-c2 * k[1:, None] * u[1:] * u[1:])

    return Gu


# to be used in MC computation
def jgetChi(dW):
    """Forcing correlation matrix multiplied by random motion
        :param dW: (dim, nPaths) random Brownian motion
        :return: (dim, nPaths) forcing applied to system
        """
    return chi_sqrt[:, None] * dW

# to be used in instanton computation
def jgetG_single(u):
    """
    :param u: (dim,) single state vector
    :return: (dim,)
    """
    Gu = jnp.zeros_like(u)
    Gu = Gu.at[1:].add(c1 * k[1:] * u[:-1] * u[:-1])
    Gu = Gu.at[:-1].add(-c1 * k[1:] * u[:-1] * u[1:])
    Gu = Gu.at[1:].add(c2 * k[1:] * u[:-1] * u[1:])
    Gu = Gu.at[:-1].add(-c2 * k[1:] * u[1:] * u[1:])
    return Gu

# to be used in instanton computation
def jgetChi_single(dW):
    """
    :param dW: (dim,)
    :return: (dim,)
    """
    return chi_sqrt * dW

def jgetF(u):
    return jnp.sum(u) # return total velocity across all shells
######################################################################
# quadrature rule for time integrals

def jgetTimeIntegral(a, b):
    ret = jnp.sum(a * b, axis = 1) * dt
    return jnp.sum(ret[:-1])

######################################################################
# JAX implementation of noise to observable map that can be automatically differentiated

def integrate_forward_jax(etaa):
    def scan_fun(u, etaaa):
        ret_u = jgetIF_single(u + dt * jgetG_single(u) + dt * jgetChi_single(etaaa), dt) # * jnp.exp(-nu * k**2 * dt)
        return ret_u, ret_u
    uT, u = jax.lax.scan(scan_fun, copy.copy(init_u), etaa[:-1])
    u = jnp.concatenate([init_u[None, :], u], axis = 0)
    return u, jgetF(uT)

def integrate_forward_obs_jax(etaa):
    return integrate_forward_jax(etaa)[1]

# jax numpy implementation of forward map that returns the full state space instanton trajectory for diagnostics
def integrate_forward(etaa):
    u = jnp.zeros((nt + 1, dim))
    u = u.at[0, :].set(init_u)
    for i in range(nt):
        u = u.at[i+1, :].set(
            jgetIF_single(u[i] + dt * jgetG_single(u[i]) + dt * jgetChi_single(etaa[i]), dt) # * jnp.exp(-nu * k**2 * dt)
        )
    return u, jgetF(u[-1])

######################################################################
# numpy implementation for direct Monte Carlo simulations of the DN shell model

def getSamplePaths(nPaths=1000):
    print('Performing direct sampling for DN shell model energy spectra')
    print('Obtaining {} simulations'.format(nPaths))

    u_current = np.zeros((dim, nPaths))

    # Store snapshots at specific time indices for plotting and statistics
    time_indices = np.linspace(0, nt, 1000).astype(int)
    u_snapshots = {}
    u_snapshots[0] = u_current.copy()

    # Euler-Maruyama steps
    for j in tqdm.tqdm(range(nt)):
        dW = np.random.randn(dim, nPaths) * np.sqrt(dt)
        u_current = jgetIF(u_current + dt * jgetG(u_current) + jgetChi(dW), dt)
        if (j + 1) in time_indices:
            u_snapshots[j + 1] = u_current.copy()

    # u_T = np.sum(u_current.copy(), axis=0)
    u_T = u_current.copy() # keeps the dimensions separate, so result is at final time T, the velocity per dimension per sample
    velo_grad = k @ u_current.copy()

    return u_snapshots, time_indices, u_T, velo_grad


######################################################################

class Instanton():
    
    def optimize(self, lbda, targetObservable = 1., mu = 0., initialEta = None):
        print('################################################')
        print("Computing Instanton for fixed penalty parameter mu = {}".format(mu))
        print("and lagrange multiplier lbda = {} and target observable = {}".format(lbda, targetObservable))
        print("Performing updates p^{k+1} = p^k - alpha (p^k - z^k)")
        print("where alpha is found via Armijo line search,")
        print("Terminates if gradient L^2 norm is below eps.")
        print("################################################")
        # initialize fields
        if initialEta is None:
            self.eta = np.random.randn(nt + 1, dim) / jnp.sqrt(dt) # initializes eta as a random white noise (Gaussian)
        else:
            self.eta = initialEta
        
        def target_func(eta):
            etaa = jnp.asarray(np.reshape(eta, (nt + 1, dim)))
            obs = integrate_forward_obs_jax(etaa)
            ret = 0.5 * jgetTimeIntegral(etaa, etaa) - lbda * (obs - targetObservable) + mu / 2. * (targetObservable - obs)**2
            return ret

        target_func_grad = jax.jacrev(target_func)
        res = spo.minimize(target_func, self.eta.flatten(), method = 'L-BFGS-B', jac = target_func_grad, options = {'ftol': 1e-8, 'gtol': 1e-8, 'disp': False})
        print(res)
        res = res.x

        self.eta = copy.copy(jnp.reshape(res, (nt + 1, dim)))
        
        u, obsValue = integrate_forward(self.eta)
        
        action = 0.5 * jgetTimeIntegral(self.eta, self.eta) # this is 1/2 * L^2 norm
        print('################################################')
        print('Parameters of the solution:')
        print('lambda =', lbda)
        print('mu =', mu)
        print('observable =', obsValue)
        print('Action =', action)
        ret = obsValue, action, lbda, copy.copy(self.eta), u
        dS = 0.5 * np.real(np.sum(self.eta**2., axis = 1) * dt)
        ret = ret + (dS,)
        return ret
        
    def searchInstantonViaAugmented(self, targetObservable, muMin = np.log10(5.), muMax = np.log10(100.), nMu = 8, initLbda = 1.):
        print("Find instanton for observable value", targetObservable, "via augmented Lagrangian method.")
        muList = np.logspace(muMin, muMax, nMu)
        obsValue, action, lbda, eta, u, dS = self.optimize(initLbda, targetObservable = targetObservable, mu =  muList[0])
        print("Mu = {}, lambda = {} yields observable = {} and action = {}".format(muList[0], lbda, obsValue, action))
        lbda = muList[0] * (targetObservable - obsValue) + initLbda
        for j in range(1, nMu):
            obsValue, action, lbda, eta, u, dS = self.optimize(lbda, targetObservable = targetObservable, mu =  muList[j], initialEta = eta)
            print("Mu = {}, lambda = {} yields observable = {} and action = {}".format(muList[j], lbda, obsValue, action))
            lbda = muList[j] * (targetObservable - obsValue) + lbda
        return obsValue, action, lbda, eta, u, dS


######################################################################
######################################################################
######################################################################

if __name__ == '__main__':

    now = datetime.now()
    dt_string = now.strftime('%Y_%m_%d_%H_%M_%S')
    data_dir = '/Users/rawdata/Downloads/data/inst/obs_{}_date_{}'.format(targetObs, dt_string)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir + '/obs.npy', targetObs)

    # uncomment for output to log file
    #sys.stdout = open('{}/output_.log'.format(data_dir), 'w', buffering = 1)

    ############################################################
    # Monte Carlo of SDE
    eps = 0.01
    u, t, u_T, velo_grad_T = getSamplePaths(nPaths=1000)

    ############################################################
    # instanton computation
    start = time.time()

    instanton = Instanton()

    obsValue, action, lbda, eta, u, dS = instanton.searchInstantonViaAugmented(targetObs)

    np.save(data_dir + '/inst_obs.npy', obsValue)
    np.save(data_dir + '/inst_act.npy', action)
    np.save(data_dir + '/inst_lbda.npy', lbda)
    np.save(data_dir + '/inst_eta.npy', eta)
    np.save(data_dir + '/inst_u.npy', u)
    np.save(data_dir + '/inst_ds.npy', dS)

    print('Needed', time.time() - start, 'seconds for the instanton computation...')

    if plots:
        plt.figure()
        plt.plot(u[:,0], u[:,1])
        plt.axvline(x = targetObs, color = 'black', linestyle = 'dashed')
        plt.scatter([x0], [y0], color = 'orange')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.savefig(data_dir + '/inst.pdf', bbox_inches = 'tight')
        plt.close()

    ############################################################
    # load previously computed instanton

    # data_dir = 'data/...'
    # obsValue = np.load(data_dir + '/inst_obs.npy')
    # action   = np.load(data_dir + '/inst_act.npy')
    # lbda     = np.load(data_dir + '/inst_lbda.npy')
    # eta      = np.load(data_dir + '/inst_eta.npy')
    # u        = np.load(data_dir + '/inst_u.npy')
    # dS       = np.load(data_dir + '/inst_ds.npy')
    # instanton = Instanton()
    
    ############################################################

    """
    print('Tail prob prefactor:', prefProb)
    np.save(data_dir + '/prefProb.npy', prefProb)
    print('Noise strength:', eps)
    print("Tail probability estimate:", np.sqrt(eps/(2. * np.pi)) * prefProb * np.exp(-action/eps))
    """
    
