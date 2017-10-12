from __future__ import absolute_import, division, print_function

__author__ = 'billhuang'

import numpy as np
import lower_bound as lb
import expectation as et
import gradient_ascent as ga
import numerical_utils as nu
import gradient as gr
import entropy

def initialization(M_, K_, hyperparams):
    params = {}
    params['alpha'] = hyperparams['aalpha'] / hyperparams['balpha']
    params['gamma'] = hyperparams['agamma'] / hyperparams['bgamma']
    params['V'] = 1 / (K_ - np.arange(K_))
    params['beta'] = sync_beta(params['V'])
    api_ = np.zeros((M_, K_))
    api_ += params['alpha'] * params['beta']
    params['mu'] = np.zeros((M_, K_))
    params['v'] = np.ones((M_, K_))
    params['Eexpnegw'] = et.expnegnormal(params['mu'], params['v'])
    bpi_ = params['Eexpnegw']
    params['Epi'] = et.gamma(api_, bpi_)
    params['xi'] = sync_xi(params['Epi'])
    params['Elnpi'] = et.loggamma(api_, bpi_)
    params['Elneta'] = nu.log(np.random.dirichlet(hyperparams['kappa'], size = K_))
    params['mean'] = sync_mean(params['mu'])
    params['Kern'] = sync_Kern(params['v'], params['mu'], params['mean'])
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
            phim[:,k] += Elneta_[k,:][Y_[m]]
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

def sync_mean(mu_):
    mean_ = np.mean(mu_, axis = 0)
    return (mean_)

def sync_Kern(v_, mu_, mean_):
    M_, K_ = mu_.shape
    Kern_ = np.zeros((K_, K_))
    for m in range(M_):
        mmu_ = mu_[m,:] - mean_
        Kern_ += np.dot(mmu_, mmu_.T) + np.diag(v_[m,:])
    Kern_ = Kern_ / M_
    return (Kern_)

def update_alpha(alpha_, aalpha_, balpha_, beta_, Elnpi_, Ew_):
    lnalpha_ = ga.ascent_step(nu.log(alpha_), gr.alpha_f, gr.alpha_fgrad,
                              aalpha_ = aalpha_, balpha_ = balpha_,
                              beta_ = beta_, Elnpi_ = Elnpi_, Ew_ = Ew_)
    alpha_ = nu.exp(lnalpha_)
    ELOB_ = lb.alpha_lower_bound(alpha_, aalpha_, balpha_)
    return (alpha_, ELOB_)

def update_gamma(agamma_, bgamma_, V_):
    K_ = V_.size
    gamma_ = (K_ - agamma_ - 2) / (bgamma_ - np.sum(nu.log(1 - V_[:-1])))
    ELOB_ = lb.gamma_lower_bound(gamma_, agamma_, bgamma_)
    return (gamma_, ELOB_)

def update_V(V_, gamma_, alpha_, Elnpi_, Ew_):
    logitV_ = nu.log(V_ / (1 - V_))
    logitV_ = ga.ascent_step(logitV_, gr.V_f, gr.V_fgrad, 
                             gamma_ = gamma_, alpha_ = alpha_,
                             Elnpi_ = Elnpi_, Ew_ = Ew_)
    V_ = 1 / (1 + np.exp(-logitV_))
    ELOB_ = lb.V_lower_bound(V_, gamma_)
    return (V_, ELOB_)

def update_pi(Y_, phi_, alpha_, beta_, Eexpnegw_, Ew_, xi_):
    M_ = len(Y_)
    K_ = beta_.size
    apipost_ = np.zeros((M_, K_))
    apipost_ += alpha_ * beta_
    bpipost_ = Eexpnegw_
    for m in range(M_):
        apipost_[m,:] += np.sum(phi_[m], axis = 0)
        bpipost_[m,:] += (Y_[m].size) / xi_[m]
    Epi_ = et.gamma(apipost_, bpipost_)
    Elnpi_ = et.loggamma(apipost_, bpipost_)
    ELOB_ = lb.pi_lower_bound(Epi_, Elnpi_, alpha_, beta_, Ew_, Eexpnegw_)
    ELOB_ -= entropy.gamma(apipost_, bpipost_)
    return (Epi_, Elnpi_, ELOB_)

def update_mu_v(mu_, v_, mean_, Kern_, alpha_, beta_, Epi_):
    M_, K_ = mu_.shape
    invKern_ = np.linalg.inv(Kern_)
    for m in range(M_):
        muv_ = np.zeros((2, K_))
        muv_[0,:] = mu_[m,:]
        muv_[1,:] = nu.log(v_[m,:])
        muv_ = ga.ascent_step(muv_, gr.muv_f, gr.muv_fgrad,
                              mean_ = mean_, invKern_ = invKern_,
                              Epim_ = Epi_[m,:],
                              alpha_ = alpha_, beta_ = beta_)
        mu_[m,:] = muv_[0,:]
        v_[m,:] = nu.exp(muv_[1,:])
    mean_ = sync_mean(mu_)
    Kern_ = sync_Kern(v_, mu_, mean_)
    Eexpnegw_ = et.expnegnormal(mu_, v_)
    ELOB_ = lb.w_lower_bound(mu_, v_, mean_, Kern_)
    ELOB_ -= entropy.normal(mu_, v_)
    return (mu_, v_, mean_, Kern_, Eexpnegw_, ELOB_)
        
def update_eta(kappa_, Y_, phi_):
    M_ = len(phi_)
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
    phi_, lb = sync_phi(Y_, params['xi'], params['Epi'], params['Elnpi'],
                        params['Elneta'])
    #print('E step')
    #print(lb)
    #print()
    ELOB_ += lb
    return (phi_, ELOB_)

def VBM(Y_, phi_, params, hyperparams):
    params['alpha'], lb1 = update_alpha(params['alpha'],
                                        hyperparams['aalpha'], hyperparams['balpha'],
                                        params['beta'], params['Elnpi'], params['mu'])
    params['gamma'], lb2 = update_gamma(hyperparams['agamma'], hyperparams['bgamma'],
                                        params['V'])
    params['V'], lb3 = update_V(params['V'], params['gamma'], params['alpha'],
                                params['Elnpi'], params['mu'])
    params['beta'] = sync_beta(params['V'])
    params['Epi'], params['Elnpi'], lb4 = update_pi(Y_, phi_,
                                                    params['alpha'], params['beta'],
                                                    params['Eexpnegw'], params['mu'],
                                                    params['xi'])
    params['xi'] = sync_xi(params['Epi'])
    params['Elneta'], lb5 = update_eta(hyperparams['kappa'], Y_, phi_)
    params['mu'], params['v'], params['mean'], params['Kern'], params['Eexpnegw'], lb6 \
                  = update_mu_v(params['mu'], params['v'],
                                params['mean'], params['Kern'],
                                params['alpha'], params['beta'], params['Epi'])
    ELOB_ = lb1 + lb2 + lb3 + lb4 + lb5 + lb6
    #print('M step')
    #print(lb1, lb2, lb3, lb4, lb5, lb6)
    #print()
    return (params, ELOB_)

def DILN_LDA(Y_, K_, hyperparams_, eps = np.power(0.1, 3)):
    print('Start Inference...')
    M_ = len(Y_)
    params_ = initialization(M_, K_, hyperparams_)
    lower_bound = np.array([])
    partial_lower_bound_ = np.nan_to_num(-np.inf)
    continue_ = True
    while (continue_):
        print('*', end = '')
        phi_, lower_bound_ = VBE(Y_, params_, partial_lower_bound_)
        #print()
        #print(lower_bound_)
        #print()
        params_, partial_lower_bound_ = VBM(Y_, phi_, params_, hyperparams_)
        lower_bound = np.append(lower_bound, lower_bound_)
        if (lower_bound.size > 2):
            if ((np.exp(lower_bound[-1] - lower_bound[-2]) - 1) < eps):
                continue_ = False
                print('  done!')
    UPi = params_['Epi']
    Pi = (UPi.T / np.sum(UPi, axis = 1)).T
    print('theta')
    print(Pi)
    
    
                                   
        
