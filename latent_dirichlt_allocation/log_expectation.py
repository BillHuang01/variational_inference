__author__ = 'billhuang'

import numpy as np
from scipy import special

def dirichlet(alpha_):
    # alpha_ is a vector
    # pi|alpha ~ Dirichlet(alpha)
    alphahat_ = np.sum(alpha_)
    lnpi_ = special.digamma(alpha_) - special.digamma(alphahat_)
    return (lnpi_)
    
