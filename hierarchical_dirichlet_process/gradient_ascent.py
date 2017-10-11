__author__ = 'billhuang'

import numpy as np

'''
def ascent_step(x, f, fgrad, gamma = 1, alpha = 0.05,
                *args, **kwargs):
    fprev_ = f(x, *args, **kwargs)
    grad_ = fgrad(x, *args, **kwargs)
    continue_ = True
    while (continue_):
        newx = x + gamma * grad_
        fcomp_ = fprev_ + alpha * gamma * np.sum(grad_ * grad_)
        if (f(newx, *args, **kwargs) < fcomp_):
            gamma = gamma * 0.8
        else:
            x = newx
            continue_ = False
    return (x)
'''
    
def ascent_step(x, f, fgrad, gamma = 1, *args, **kwargs):
    fprev_ = f(x, *args, **kwargs)
    grad_ = fgrad(x, *args, **kwargs)
    continue_ = True
    while (continue_):
        newx = x + gamma * grad_
        if (f(newx, *args, **kwargs) < fprev_):
            gamma = gamma * 0.8
        else:
            x = newx
            continue_ = False
    return (x)
