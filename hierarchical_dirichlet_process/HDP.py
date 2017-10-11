from __future__ import absolute_import, division, print_function

__author__ = 'billhuang'

import numpy as np
import lower_bound as lb
import expectation as et
import gradient_ascent as ga
import gradient as gr
import numerical_utils as nu
import entropy

def initialization(M_, K_, hyperparams):
    params = {}
    params['alpha'] = hyperparams['aalpha'] / hyperparams['balpha']
    params['gamma'] = hyperparams['agamma'] / hyperparams['bgamma']
    params['V'] = 1 / (K_ - np.arange(K_))
    params['beta'] = sync_beta(params['V'])
    api_ = np.zeros((M_, K_))
    api_ += params['alpha'] * params['beta']
    bpi_ = np.ones((M_, K_))
    params['Epi'] = et.gamma(api_, bpi_)
    params['xi'] = sync_xi(params['Epi'])
    params['Elnpi'] = et.loggamma(api_, bpi_)
    params['Elneta'] = nu.log(np.random.dirichlet(hyperparams['kappa'], size = K_))
    return (params)

def sync_phi(Y_, xi_, Epi_, Elnpi_, Elneta_):
    M_, K_ = Elnpi_.shape
    phi_ = []
    ELOB_ = 0
    for m in range(M_):
        Ym_ = Y_[m]
        phim = np.zeros((Ym_.size, K_))
        phim += Elnpi_[m,:]
        for k in range(K_):
            phim[:,k] += Elneta_[k,:][Ym_]
        phim_ = nu.normalize_log_across_row(phim)
        phi_.append(phim_)
        ELOB_ += lb.data_lower_bound(Ym_, phim_, Elneta_)
        ELOB_ += lb.C_lower_bound(phim_, Elnpi_[m,:], Epi_[m,:], xi_[m])
        ELOB_ -= entropy.discrete(phim_)
    return (phi_, ELOB_)

def sync_xi(Epi_):
    xi_ = np.sum(Epi_, axis = 1)
    return (xi_)

def sync_beta(V_):
    beta_ = V_ * (np.append(1, np.cumprod(1 - V_[:-1])))
    return (beta_)

def update_alpha(alpha_, aalpha_, balpha_, Elnpi_, beta_):
    lnalpha_ = ga.ascent_step(nu.log(alpha_), gr.alpha_f, gr.alpha_fgrad,
                              aalpha_ = aalpha_, balpha_ = balpha_,
                              Elnpi_ = Elnpi_, beta_ = beta_)
    alpha_ = np.exp(lnalpha_)
    ELOB_ = lb.alpha_lower_bound(alpha_, aalpha_, balpha_)
    return (alpha_, ELOB_)

def update_gamma(agamma_, bgamma_, V_):
    K_ = V_.size
    gamma_ = (K_ - agamma_ - 2) / (bgamma_ - np.sum(nu.log(1 - V_[:-1])))
    ELOB_ = lb.gamma_lower_bound(gamma_, agamma_, bgamma_)
    return (gamma_, ELOB_)

def update_V(V_, gamma_, alpha_, Elnpi_):
    logitV = nu.log(V_ / (1 - V_))
    logitV_ = ga.ascent_step(logitV, gr.V_f, gr.V_fgrad, 
                             gamma_ = gamma_, alpha_ = alpha_,
                             Elnpi_ = Elnpi_)
    V_ = 1 / (1 + np.exp(-logitV_))
    ELOB_ = lb.V_lower_bound(V_, gamma_)
    return (V_, ELOB_)

def update_pi(Y_, phi_, alpha_, beta_, xi_):
    M_ = len(Y_)
    K_ = beta_.size
    apipost_ = np.zeros((M_, K_))
    apipost_ += alpha_ * beta_
    bpipost_ = np.ones((M_, K_))
    for m in range(M_):
        apipost_[m,:] += np.sum(phi_[m], axis = 0)
        bpipost_[m,:] += (Y_[m].size)/(xi_[m])
    Epi_ = et.gamma(apipost_, bpipost_)
    Elnpi_ = et.loggamma(apipost_, bpipost_)
    ELOB_ = lb.pi_lower_bound(Epi_, Elnpi_, alpha_, beta_)
    ELOB_ -= entropy.gamma(apipost_, bpipost_)
    return (Epi_, Elnpi_, ELOB_)

def update_eta(kappa_, Y_, phi_):
    M_ = len(Y_)
    K_ = phi_[0].shape[1]
    T_ = kappa_.size
    Elneta_ = np.zeros((K_, T_))
    ELOB_ = 0
    for k in range(K_):
        kappapost_ = np.zeros(T_)
        for m in range(M_):
            for i in range(Y_[m].size):
                kappapost_[Y_[m][i]] += phi_[m][i,k]
        kappapost_ += kappa_
        Elneta_[k,:] = et.logdirichlet(kappapost_)
        ELOB_ += lb.eta_lower_bound(Elneta_[k,:], kappa_)
        ELOB_ -= entropy.dirichlet(kappapost_)
    return (Elneta_, ELOB_)

def VBE(Y_, params, ELOB_):
    phi_, lb = sync_phi(Y_, params['xi'], params['Epi'],
                        params['Elnpi'], params['Elneta'])
    ELOB_ += lb
    return (phi_, ELOB_)

def VBM(Y_, phi_, params, hyperparams):
    params['alpha'], lb1 = update_alpha(params['alpha'],
                                        hyperparams['aalpha'], hyperparams['balpha'],
                                        params['Elnpi'], params['beta'])
    params['gamma'], lb2 = update_gamma(hyperparams['agamma'], hyperparams['bgamma'],
                                        params['V'])
    params['V'], lb3 = update_V(params['V'],
                                params['gamma'], params['alpha'], params['Elnpi'])
    params['beta'] = sync_beta(params['V'])
    params['Epi'], params['Elnpi'], lb4 = update_pi(Y_, phi_,
                                                    params['alpha'], params['beta'],
                                                    params['xi'])
    params['xi'] = sync_xi(params['Epi'])
    params['Elneta'], lb5 = update_eta(hyperparams['kappa'], Y_, phi_)
    ELOB_ = lb1 + lb2 + lb3 + lb4 + lb5
    return (params, ELOB_)

def HDP(Y_, K_, hyperparams, eps = np.power(0.1,3)):
    print('Start Inference...')
    params = initialization(len(Y_), K_, hyperparams)
    lower_bound = np.array([])
    partial_ELOB_ = np.nan_to_num(-np.inf)
    continue_ = True
    while (continue_):
        print('*', end = '')
        phi_, ELOB_ = VBE(Y_, params, partial_ELOB_)
        lower_bound = np.append(lower_bound, ELOB_)
        params, partial_ELOB_ = VBM(Y_, phi_, params, hyperparams)
        if (lower_bound.size > 2):
            if ((np.exp(lower_bound[-1] - lower_bound[-2]) - 1) < eps):
                continue_ = False
                print('  done!')
    UPi = params['Epi']
    Pi = (UPi.T / np.sum(UPi, axis = 1)).T
    print('theta')
    print(Pi)
    
