__author__ = 'billhuang'

import numpy as np
import numerical_utils as nu
from scipy.special import gammaln
from scipy import stats

def data_lower_bound(Ym_, phim_, Elneta_):
    lower_bound_ = np.sum(phim_.T * Elneta_[:,Ym_])
    return (lower_bound_)

def C_lower_bound(phim_, Elnpim_, Epim_, xim_):
    sub_ = -nu.log(xim_) - (np.sum(Epim_) - xim_) / (xim_)
    NElnpim_ = Elnpim_ + sub_
    lower_bound_ = np.sum(phim_ * NElnpim_)
    return (lower_bound_)

def pi_lower_bound(Epi_, Elnpi_, alpha_, beta_, Ew_, Eexpnegw_):
    M_ = Epi_.shape[0]
    lower_bound_ = np.sum((alpha_ * beta_ - 1) * Elnpi_)
    lower_bound_ -= np.sum(Eexpnegw_ * Epi_)
    lower_bound_ -= np.sum(alpha_ * beta_ * Ew_)
    lower_bound_ -= M_ * np.sum(gammaln(alpha_ * beta_))
    return (lower_bound_)

def V_lower_bound(V_, gamma_):
    lower_bound_ = np.sum(nu.log(gamma_) + (gamma_-1) * nu.log(1 - V_[:-1]))
    return (lower_bound_)

def eta_lower_bound(Elnetak_, kappa_):
    lower_bound_ = gammaln(np.sum(kappa_)) - np.sum(gammaln(kappa_))
    lower_bound_ += np.sum((kappa_ - 1) * Elnetak_)
    return (lower_bound_)

def w_lower_bound(mu_, v_, mean_, Kern_):
    lower_bound_ = np.sum(stats.multivariate_normal.logpdf(mu_, mean_, Kern_))
    diaginvKern_ = np.diag(np.linalg.inv(Kern_))
    lower_bound_ -= 0.5 * np.sum(np.dot(v_, diaginvKern_))
    return (lower_bound_)

def alpha_lower_bound(alpha_, a_, b_):
    lower_bound_ = (a_ - 1) * nu.log(alpha_) - b_ * alpha_ + a_ * nu.log(b_) - gammaln(a_)
    return (lower_bound_)

def gamma_lower_bound(gamma_, a_, b_):
    lower_bound_ = (a_ - 1) * nu.log(gamma_) - b_ * gamma_ + a_ * nu.log(b_) - gammaln(a_)
    return (lower_bound_)


    
