__author__ = 'billhuang'

import numpy as np
import lower_bound as lb
import entropy
import expectation as et
import numerical_utils as nu
from scipy.special import digamma, gammaln

def alpha_f(lnalpha_, aalpha_, balpha_, beta_, Elnpi_, Ew_):
    alpha_ = np.exp(lnalpha_)
    f_ = (aalpha_ - 1) * lnalpha_ - balpha_ * alpha_
    abeta_ = alpha_ * beta_
    f_ += np.sum(abeta_ * (Elnpi_ - Ew_) - gammaln(abeta_))
    return (f_)

def alpha_fgrad(lnalpha_, aalpha_, balpha_, beta_, Elnpi_, Ew_):
    # use exponential trick since alpha_ >= 0
    alpha_ = np.exp(lnalpha_)
    grad_ = (aalpha_ - 1) / alpha_ + balpha_
    grad_ += np.sum(beta_ * (Elnpi_ - Ew_ - digamma(alpha_ * beta_)))
    grad_ *= alpha_
    return (grad_)

def V_f(logitV_, gamma_, alpha_, Elnpi_, Ew_):
    V_ = 1 / (1 + np.exp(-logitV_))
    beta_ = V_ * (np.append(1, np.cumprod(1 - V_[:-1])))
    f_ = (gamma_ - 1) * np.sum(nu.log(1 - V_[:-1]))
    abeta_ = alpha_ * beta_
    f_ += np.sum(abeta_ * (Elnpi_ - Ew_) - gammaln(abeta_))
    return (f_)

def V_fgrad(logitV_, gamma_, alpha_, Elnpi_, Ew_):
    # use logit trick since 0 <= V <= 1
    M_, K_ = Elnpi_.shape
    expneglogitV_ = np.exp(-logitV_)
    V_ = 1 / (1 + expneglogitV_)
    one_V_ = 1 - V_
    leftprob_ = np.append(1, np.cumprod(one_V_[:-1]))
    beta_ = V_ * leftprob_
    psiabeta_ = digamma(alpha_ * beta_)
    grad_ = alpha_ * leftprob_ * np.sum((Elnpi_ - Ew_ - psiabeta_), axis = 0)
    partg_ = -alpha_ * beta_ * np.sum((Elnpi_ - Ew_ - psiabeta_), axis = 0)
    for k in range(K_ - 1):
        grad_[k] += np.sum(partg_[(k+1):]) / (one_V_[k])
    grad_[:-1] -= (gamma_ - 1) / (one_V_[:-1])
    grad_ *= expneglogitV_ / np.square(1 + expneglogitV_)
    grad_[-1] = 0
    return (grad_)

def muv_f(muv_, mean_, invKern_, Epim_, alpha_, beta_):
    mu_ = muv_[0,:]
    v_ = nu.exp(muv_[1,:])
    Eexpnegw_ = np.exp(-mu_ + 0.5 * v_)
    f_ = np.sum(Epim_ * Eexpnegw_) - np.sum(mu_ * alpha_ * beta_)
    mmu_ = mu_ - mean_
    f_ -= 0.5 * np.dot(np.dot(mmu_.T, invKern_), mmu_)
    f_ -= 0.5 * np.sum(np.diag(invKern_) * v_)
    f_ += 0.5 * np.sum(nu.log(v_))
    return (f_)

def muv_fgrad(muv_, mean_, invKern_, Epim_, alpha_, beta_):
    mu_ = muv_[0,:]
    v_ = nu.exp(muv_[1,:])
    Eexpnegw_ = nu.exp(-mu_ + 0.5 * v_)
    grad_ = np.zeros(muv_.shape)
    grad_[0,:] = Epim_ * Eexpnegw_ - alpha_ * beta_ - np.dot(invKern_, (mu_ - mean_))
    grad_[1,:] = 0.5 * (-Epim_ * Eexpnegw_ + (1/v_) - np.diag(invKern_)) * v_
    return (grad_)
    

