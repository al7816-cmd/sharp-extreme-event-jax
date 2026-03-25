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

######################################################################
# set up parameters for model

T = 10
dt = 1e-4
nt = int(T / dt)
t = jnp.linspace(0., T, nt + 1)

# DN shell model parameters
dim = 17    # number of discretized concentric shells n = 0 ... dim - 1
c1 = 1.0    # constant parameter for getG
c2 = 0.0    # constant parameter for getG
nu = 1e-6   # viscosity
k = 2 ** jnp.linspace(0, dim-1, dim) # wavenumbers
sigma = 1.  # noise strength for direct sampling
chi_sqrt = sigma * k**(-3.)  # forcing cov sqrt

# initial condition: start at fixed point
x0 = 1. / (5. * jnp.sqrt(2.))
y0 = x0 + 0.2
init_phi = jnp.array([x0, y0])

targetObs = 0.5

print('Running Instanton computation for DN shell model')
print(f'Parameters: T={T}, dt={dt}, sigma^2={sigma ** 2.}, dim={dim}')


plots = True # option for producing plots of instanton and operator spectra
projectEtaPerp = True # set to false for calculating MGF prefactor and explicit transformation to tail probability; only works for convex rate function
######################################################################
# define model functions

getIF = lambda u, dt: u * jnp.exp(-nu * k[:, None]**2 * dt)  # integrating factor for linear terms in b times x

getEnergy = lambda u: 0.5 * u**2
getEnergyDissipation = lambda u : jnp.sum(nu * k[:,None]**2 * u**2, axis = 0)

def jgetG(u):
    """
    :param u: (dim, nPaths)
    :return: (dim, nPaths)
    """
    Gu = jnp.zeros_like(u)

    Gu = Gu.at[1:].add(c1 * k[1:, None] * u[:-1] * u[:-1])
    Gu = Gu.at[:-1].add(-c1 * k[1:, None] * u[:-1] * u[1:])
    Gu = Gu.at[1:].add(c2 * k[1:, None] * u[:-1] * u[1:])
    Gu = Gu.at[:-1].add(-c2 * k[1:, None] * u[1:] * u[1:])

    return Gu

def jgetChi(dW):
    """Forcing correlation matrix multiplied by random motion
        :param dW: (dim, nPaths) random Brownian motion
        :return: (dim, nPaths) forcing applied to system
        """
    return chi_sqrt[:, None] * dW


######################################################################
# quadrature rule for time integrals

def jgetTimeIntegral(a, b):
    ret = jnp.sum(a * b, axis = 1) * dt
    return jnp.sum(ret[:-1])

######################################################################
# JAX implementation of noise to observable map that can be automatically differentiated

def integrate_forward_jax(etaa):
    def scan_fun(phi, etaaa):
        ret_phi = phi + dt * (jgetG(phi) + jgetSigma(phi, etaaa))
        return ret_phi, ret_phi
    phiT, phi = jax.lax.scan(scan_fun, copy.copy(init_phi), etaa[:-1])
    phi = jnp.concatenate([init_phi[None, :], phi], axis = 0)
    return phi, jgetF(phiT)

def integrate_forward_obs_jax(etaa):
    return integrate_forward_jax(etaa)[1]

# jax numpy implementation of forward map that returns the full state space instanton trajectory for diagnostics
def integrate_forward(etaa):
    phi = jnp.zeros((nt + 1, dim))
    phi = phi.at[0,:].set(jnp.array([x0, y0]))
    for i in range(nt):
        phi = phi.at[i+1,:].set(phi[i] + dt * \
             (jgetB(phi[i]) + jgetSigma(phi[i], etaa[i])))
    return phi, jgetF(phi[-1])

# vectorized numpy implementation for direct Monte Carlo simulations of SDE
def getSamples(nParallel = int(1e4), maxSim = int(1e6), eps = 0.01, conf = 0.95):
    print('################################################')
    print('Performing direct sampling with eps = {}'.format(eps))
    print('Copmuting {} samples, with {} SDE simluations in parallel'.format(maxSim, nParallel))
    obss = np.zeros((maxSim,))
    nSamples = 0
    for run in tqdm.tqdm(range(maxSim//nParallel)):
        x = copy.copy(init_phi)[:, None]
        for j in range(nt):
            x = x + dt * jgetB(x) + np.sqrt(eps * dt) \
                        * jgetSigma(x, np.random.randn(dim, nParallel))
        obss[nSamples:(nSamples+nParallel)] = jgetF(x)
        nSamples += nParallel
    print('')
    print('Completed MC {} simulations with mean observable {}+-{}'.format(nSamples, np.mean(obss), np.std(obss)))

    prob = np.sum(np.where(obss >= targetObs, 1., 0.)) / nSamples
    std  = np.sqrt(prob * (1. - prob) /nSamples)
    c = norm.ppf((conf + 1) / 2.)
    print('')
    print('Estimated tail probability from this data: {}'.format(prob))
    print('Asymptotic {}% confidence interval: [{},{}]'.format( \
            100 * conf, round(prob - c * std, int(-np.log10(prob) + 3)), \
            round(prob + c * std, int(-np.log10(prob) + 3))))
    print('################################################')
    return obss

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
            self.eta = np.random.randn(nt + 1, dim)**2 * 0.1 + 0.2
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
        
        phi, obsValue = integrate_forward(self.eta)
        
        action = 0.5 * jgetTimeIntegral(self.eta, self.eta)
        print('################################################')
        print('Parameters of the solution:')
        print('lambda =', lbda)
        print('mu =', mu)
        print('observable =', obsValue)
        print('Action =', action)
        ret = obsValue, action, lbda, copy.copy(self.eta), phi
        dS = 0.5 * np.real(np.sum(self.eta**2., axis = 1) * dt)
        ret = ret + (dS,)
        return ret
        
    def searchInstantonViaAugmented(self, targetObservable, muMin = np.log10(5.), muMax = np.log10(100.), nMu = 8, initLbda = 1.):
        print("Find instanton for observable value", targetObservable, "via augmented Lagrangian method.")
        muList = np.logspace(muMin, muMax, nMu)
        obsValue, action, lbda, eta, phi, dS = self.optimize(initLbda, targetObservable = targetObservable, mu =  muList[0])
        print("Mu = {}, lambda = {} yields observable = {} and action = {}".format(muList[0], lbda, obsValue, action))
        lbda = muList[0] * (targetObservable - obsValue) + initLbda
        for j in range(1, nMu):
            obsValue, action, lbda, eta, phi, dS = self.optimize(lbda, targetObservable = targetObservable, mu =  muList[j], initialEta = eta)
            print("Mu = {}, lambda = {} yields observable = {} and action = {}".format(muList[j], lbda, obsValue, action))
            lbda = muList[j] * (targetObservable - obsValue) + lbda
        return obsValue, action, lbda, eta, phi, dS


######################################################################
######################################################################
######################################################################

if __name__ == '__main__':

    now = datetime.now()
    dt_string = now.strftime('%Y_%m_%d_%H_%M_%S')
    data_dir = 'data/obs_{}_date_{}'.format(targetObs, dt_string)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir + '/obs.npy', targetObs)

    # uncomment for output to log file
    #sys.stdout = open('{}/output_.log'.format(data_dir), 'w', buffering = 1)

    ############################################################
    # Monte Carlo of SDE
    eps = 0.01
    getSamples(nParallel = int(1e4), maxSim = int(5.2e5), eps = eps, conf = 0.95)

    ############################################################
    # instanton computation
    start = time.time()

    instanton = Instanton()

    obsValue, action, lbda, eta, phi, dS = instanton.searchInstantonViaAugmented(targetObs)

    np.save(data_dir + '/inst_obs.npy', obsValue)
    np.save(data_dir + '/inst_act.npy', action)
    np.save(data_dir + '/inst_lbda.npy', lbda)
    np.save(data_dir + '/inst_eta.npy', eta)
    np.save(data_dir + '/inst_phi.npy', phi)
    np.save(data_dir + '/inst_ds.npy', dS)

    print('Needed', time.time() - start, 'seconds for the instanton computation...')

    if plots:
        plt.figure()
        plt.plot(phi[:,0], phi[:,1])
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
    # phi      = np.load(data_dir + '/inst_phi.npy')
    # dS       = np.load(data_dir + '/inst_ds.npy')
    # instanton = Instanton()
    
    ############################################################

    print('Tail prob prefactor:', prefProb)
    np.save(data_dir + '/prefProb.npy', prefProb)
    print('Noise strength:', eps)
    print("Tail probability estimate:", np.sqrt(eps/(2. * np.pi)) * prefProb * np.exp(-action/eps))
    
