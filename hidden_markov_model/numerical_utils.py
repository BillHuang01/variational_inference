__author__ = 'billhuang'

import numpy as np

def log(x):
    '''
    safe log for handling the case with zero count
    '''
    return (np.nan_to_num(np.log(x)))

def log_matrix_multiply_vector(logM, logv):
    logNM = logM + logv
    logP = np.logaddexp.reduce(logNM, axis = 1)
    return (logP)

def normalize_across_row(M):
    sum_ = np.sum(M, axis = 1)
    nM = (M.T / sum_).T
    return (nM)
