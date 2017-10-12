__author__ = 'billhuang'

import numpy as np

def exp(x):
    return (np.nan_to_num(np.exp(x)))

def log(x):
    '''
    safe log for handling the case with zero count
    '''
    return (np.nan_to_num(np.log(x)))

def normalize_log_across_row(logM):
    '''
    safer way to convert log probability to normalized
    probability across row in a matrix than simply exponential
    each term and then normalize
    '''
    logP = np.logaddexp.reduce(logM, axis = 1)
    NlogM = (logM.T - logP).T
    M = np.exp(NlogM)
    return (M)
