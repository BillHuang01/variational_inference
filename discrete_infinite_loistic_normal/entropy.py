import numpy as np
import expectation
import numerical_utils as nu
from scipy.special import digamma, gammaln

def beta(a, b):
    '''
    beta is a special dirichlet with alpha being size 2
    '''
    entropy = dirichlet(np.array([a,b]))
    return (entropy)

def gamma(a, b):
    '''
    pi|a,b ~ Gamma(a,b)
    '''
    entropy = np.sum((a-1)*digamma(a) - a + nu.log(b) - gammaln(a))
    return (entropy)

def normal(mu, v):
    '''
    w|mu, v ~ N(mu,v)
    '''
    entropy = np.sum(-0.5 * nu.log(2 * np.pi * np.e * v))
    return (entropy)

def dirichlet(alpha):
    '''
    pi|alpha ~ Dirichlet(alpha)
    '''
    Elnpi = expectation.logdirichlet(alpha)
    entropy = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))
    entropy += np.sum((alpha - 1) * Elnpi)
    return (entropy)

def discrete(pi):
    '''
    x|pi ~ Discrete(pi)
    '''
    entropy = np.sum(pi * nu.log(pi))
    return (entropy)
