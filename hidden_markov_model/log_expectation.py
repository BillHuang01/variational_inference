__author__ = 'billhuang'

import numpy as np
from scipy import special

def dirichlet(alpha_):
    # alpha_ is a vector
    # pi|alpha ~ Dirichlet(alpha)
    alphahat_ = np.sum(alpha_)
    lnpi_ = special.digamma(alpha_) - special.digamma(alphahat_)
    return (lnpi_)

def gamma(a, b):
    # a, b must be the same dimension
    # pi|a,b ~ Gamma(a,b)
    lnpi_ = special.digamma(a) - np.log(b)
    return (lnpi_)
    
