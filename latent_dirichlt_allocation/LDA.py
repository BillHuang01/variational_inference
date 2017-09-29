from __future__ import absolute_import, division, print_function

# Latent Dirichlet Allocation
# Mixed Membership Modeling
# Topic Modeling
# Variational Inference

__author__ = 'billhuang'

import numpy as np
import entropy
import numerical_utils as nu
import log_expectation as le
import sys

def random_initialization(K_, D_, alpha_, gamma_):
    theta_ = nu.log(np.random.dirichlet(alpha_, size = D_))
    beta_ = nu.log(np.random.dirichlet(gamma_, size = K_))
    return (theta_, beta_)

def sync_phi(Y_, theta_, beta_, K_, D_):
    phi_ = []
    lower_bound_ = 0
    for d in range(D_):
        YD_ = Y_[d].size
        phid = np.zeros((YD_, K_))
        phid += theta_[d,:]
        for k in range(K_):
            phid[:,k] += beta_[k,:][Y_[d]]
        phid_ = nu.normalize_log_across_row(phid)
        phi_.append(phid_)
        lower_bound_ += np.sum(phid_ * theta_[d,:])
        lower_bound_ -= np.sum(phid_ * nu.log(phid_))
    return (phi_, lower_bound_)

def update_theta(phi_, alpha_, K_, D_):
    theta_ = np.zeros((D_, K_))
    lower_bound_ = 0
    for d in range(D_):
        alphapost_ = alpha_ + np.sum(phi_[d], axis = 0)
        theta_[d,:] = le.dirichlet(alphapost_)
        lower_bound_ += entropy.dirichlet(alpha_, theta_[d,:])
        lower_bound_ -= entropy.dirichlet(alphapost_, theta_[d,:])
    return (theta_, lower_bound_)

def update_beta(Y_, phi_, gamma_, K_, D_, V_):
    beta_ = np.zeros((K_, V_))
    lower_bound_ = 0
    for k in range(K_):
        gammapost_ = np.zeros(V_)
        for d in range(D_):
            for i in range(Y_[d].size):
                gammapost_[Y_[d][i]] += phi_[d][i,k]
        gammapost_ += gamma_
        beta_[k,:] = le.dirichlet(gammapost_)
        lower_bound_ += entropy.dirichlet(gamma_, beta_[k,:])
        lower_bound_ -= entropy.dirichlet(gammapost_, beta_[k,:])
    return (beta_, lower_bound_)

def likelihood_bound(Y_, phi_, theta_, beta_, K_, D_):
    lower_bound_ = 0
    for d in range(D_):
        for k in range(K_):
            lower_bound_ += np.sum(phi_[d][:,k] * beta_[k,:][Y_[d]])
    return (lower_bound_)

def VBE(Y_, K_, D_, theta_, beta_, lower_bound_):
    phi_, l3_ = sync_phi(Y_, theta_, beta_, K_, D_)
    l4_ = likelihood_bound(Y_, phi_, theta_, beta_, K_, D_)
    lower_bound_ += l3_ + l4_
    return (phi_, lower_bound_)

def VBM(Y_, K_, D_, V_, phi_, alpha_, gamma_):
    theta_, lb1_ = update_theta(phi_, alpha_, K_, D_)
    beta_, lb2_ = update_beta(Y_, phi_, gamma_, K_, D_, V_)
    lower_bound_ = lb1_ + lb2_
    return (theta_, beta_, lower_bound_)

def LDA(Y_, K_, D_, V_, eps = np.power(0.1, 3)):
    # K_ = # topics
    # V_ = # unique words
    # D_ = # documents
    print('Start Inference...')
    alpha_ = np.ones(K_)
    gamma_ = np.ones(V_)
    theta_, beta_ = random_initialization(K_, D_, alpha_, gamma_)
    lower_bound = np.array([])
    partial_lower_bound_ = np.nan_to_num(-np.inf)
    continue_ = True
    while (continue_):
        sys.stdout.write('|')
        phi_, lower_bound_ = VBE(Y_, K_, D_, theta_, beta_, partial_lower_bound_)
        theta_, beta_, partial_lower_bound_ = VBM(Y_, K_, D_, V_, phi_, alpha_, gamma_)
        lower_bound = np.append(lower_bound, lower_bound_)
        if (lower_bound.size > 2):
            if ((np.exp(lower_bound[-1] - lower_bound[-2]) - 1) < eps):
                continue_ = False
                sys.stdout.write('  done!\n')
    print('theta')
    print(np.exp(theta_))
    print('beta')
    print(np.exp(beta_))


        
        
    
