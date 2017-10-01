from __future__ import absolute_import, division, print_function

# Hidden Markov Model with Categorical Distribution

__author__ = 'billhuang'

import numpy as np
import entropy
import numerical_utils as nu
import log_expectation as le
from scipy import special

def random_initialization(Y_, J_):
    C_ = np.unique(Y_).size
    pi0_ = nu.log(np.random.dirichlet(np.ones(J_)))
    A_ = nu.log(np.random.dirichlet(np.ones(J_), size = J_))
    Bstar_ = nu.log(np.random.dirichlet(np.ones(C_), size = J_))
    return (pi0_, A_, Bstar_)
    
def sync_B(Y_, Bstar_):
    BstarT_ = Bstar_.T
    B_ = BstarT_[Y_, :]
    return (B_)

def pass_message_forward(pi0_, A_, B_):
    T_, K_ = B_.shape
    M_ = np.zeros((T_, K_))
    M_[0,:] = pi0_ + B_[0,:]
    for t in range(1, T_):
        M_[t,:] = nu.log_matrix_multiply_vector(A_.T, M_[(t-1),:]) + B_[t,:]
    return (M_)

def pass_message_backward(A_, B_):
    R_ = np.zeros(B_.shape)
    R_[-1,:] = 0
    for t in range(B_.shape[0] - 2, -1, -1):
        R_[t,:] = nu.log_matrix_multiply_vector(A_, (B_[(t+1),:] + R_[(t+1),:]))
    return (R_)

def sync_Q(M_, R_):
    Qu_ = M_ + R_
    logQ_ = (Qu_.T - np.logaddexp.reduce(Qu_, axis = 1)).T
    Q_ = np.exp(logQ_)
    return (Q_)

def sync_N(M_, R_, A_, B_):
    T_, K_ = B_.shape
    xi_ = np.zeros((T_ - 1, K_, K_))
    for t in range(0, T_ - 1):
        xi_[t,:] = (A_.T + M_[t,:]).T + B_[(t+1),:] + R_[(t+1),:]
    xi_ = xi_ - np.logaddexp.reduce(M_[-1,:])
    N_ = np.sum(np.exp(xi_), axis = 0)
    return (N_)

def sync_pi0(q0_, ipi0_):
    dpi0_ = q0_ + ipi0_
    pi0_ = le.dirichlet(dpi0_)
    lower_bound_ = entropy.dirichlet(ipi0_, pi0_)
    lower_bound_ -= entropy.dirichlet(dpi0_, pi0_)
    return (pi0_, lower_bound_)

def sync_A(N_, ia0_):
    J_ = N_.shape[0]
    A_ = np.zeros((N_.shape))
    da_ = N_ + ia0_
    lower_bound_ = 0
    for j in range(J_):
        A_[j,:] = le.dirichlet(da_[j,:])
        lower_bound_ += entropy.dirichlet(ia0_, A_[j,:])
        lower_bound_ -= entropy.dirichlet(da_[j,:], A_[j,:])
    return (A_, lower_bound_)

def update_params(Y_, Q_, ib0_):
    C_ = np.unique(Y_).size
    J_ = Q_.shape[1]
    Bcount_ = np.zeros((J_, C_))
    Bstar_ = np.zeros((J_, C_))
    for c in range(C_):
        Bcount_[:,c] = np.sum(Q_[(Y_==c),:], axis = 0)
    db_ = Bcount_ + ib0_
    lower_bound_ = 0
    for j in range(J_):
        Bstar_[j,:] = le.dirichlet(db_[j,:])
        lower_bound_ += entropy.dirichlet(ib0_, Bstar_[j,:])
        lower_bound_ -= entropy.dirichlet(db_[j,:], Bstar_[j,:])
    return (Bstar_, lower_bound_)

def state_bound(M_):
    #bound_ = np.sum(np.logaddexp.reduce(M_, axis = 1))
    bound_ = np.logaddexp.reduce(M_[-1,:])
    return (bound_)

def VBE(Y_, pi0_, A_, Bstar_, lower_bound_):
    B_ = sync_B(Y_, Bstar_)
    M_ = pass_message_forward(pi0_, A_, B_)
    lower_bound_ += state_bound(M_)
    R_ = pass_message_backward(A_, B_)
    Q_ = sync_Q(M_, R_)
    N_ = sync_N(M_, R_, A_, B_)
    return (Q_, N_, lower_bound_)

def VBM(Y_, Q_, N_, ipi0_, ia0_, ib0_):
    pi0_, lb1_ = sync_pi0(Q_[0,:], ipi0_)
    A_, lb2_ = sync_A(N_, ia0_)
    Bstar_, lb3_ = update_params(Y_, Q_, ib0_)
    lower_bound_ = lb1_ + lb2_ + lb3_
    return (pi0_, A_, Bstar_, lower_bound_)

def HMM(Y_, J_, eps = np.power(0.1, 3)):
    print('Start Inference...')
    ipi0_ = np.ones(J_)
    ia0_ = np.ones(J_)
    ib0_ = np.ones(np.unique(Y_).size)
    pi0_, A_, Bstar_ = random_initialization(Y_, J_)
    lower_bound = np.array([])
    partial_lower_bound_ = np.nan_to_num(-np.inf)
    continue_ = True
    while (continue_):
        print('*', end = '')
        Q_, N_, lower_bound_ = VBE(Y_, pi0_, A_, Bstar_, partial_lower_bound_)
        pi0_, A_, Bstar_, partial_lower_bound_ = VBM(Y_, Q_, N_, ipi0_, ia0_, ib0_)
        lower_bound = np.append(lower_bound, lower_bound_)
        if (lower_bound.size > 2):
            if ((np.exp(lower_bound[-1] - lower_bound[-2]) - 1) < eps):
                continue_ = False
                print('  done!')
    print('A')
    print(A_)
    print('B')
    print(Bstar_)
    
