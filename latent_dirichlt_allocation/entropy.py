import numpy as np
from scipy.special import gammaln

def dirichlet(alpha, Elnmu):
    '''
    mu|alpha ~ Dirichlet(alpha)
    '''
    entropy = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))
    entropy += np.sum((alpha - 1) * Elnmu)
    return (entropy)
