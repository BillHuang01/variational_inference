__author__ = 'billhuang'

import numpy as np
import numerical_utils as nu
from scipy.special import digamma

def gamma(a, b):
    # pi|a,b ~ Gamma(a,b)
    pi_ = a / b
    return (pi_)

def loggamma(a, b):
    # pi|a,b ~ Gamma(a,b)
    lnpi_ = digamma(a) - nu.log(b)
    return (lnpi_)

def expnegnormal(mu, v):
    # w|mu, v ~ N(mu, v)
    expnegw_ = nu.exp(-mu + 0.5 * v)
    return (expnegw_)

def logdirichlet(alpha_):
    # alpha_ is a vector
    # pi|alpha ~ Dirichlet(alpha)
    alphahat_ = np.sum(alpha_)
    lnpi_ = digamma(alpha_) - digamma(alphahat_)
    return (lnpi_)
