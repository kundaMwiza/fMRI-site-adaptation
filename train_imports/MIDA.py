# Copyright (c) 2019 Mwiza Kunda

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import numpy as np
from numpy.linalg import multi_dot
from sklearn.metrics.pairwise import pairwise_kernels

# rbf width paramter
def width_rbf(X):
    n = X.shape[0]
    Xmed = X

    G = np.sum(Xmed*Xmed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )

    dists = Q + R - 2* np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_x = np.sqrt( 0.5 * np.median(dists[dists>0]))
    
    return width_x

# rbf kernel matrix
def rbf_dot(pattern1, pattern2, deg):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
    H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)

    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    H = Q + R - 2* np.dot(pattern1, pattern2.T)

    H = np.exp(-H/2/(deg**2))
    return H

# mask out test labels
def label_information(Y, test_ind):
    Y_cp = Y.copy()
    Y_cp[test_ind, :] = [0,0]
    return Y_cp

# Maximum Independene Domain Adaptation
def MIDA(X, D, Y=None, mu = 0.1, gamma_y=0.1, h=1035, labels=False):
    """
        X  : All subjects feature matrix (n x m)
        D  : domain feature matrix (n x num_domains)
        Y  : label information matrix (n x 2)
        mu : covariance maximisation parameter
        gamma_y : dependence on label information paramter
        h  : dimensionality of projected samples
    return:
        Z  : projected samples
    """

    # # Augment features with domain information
    X = np.concatenate([X, D], axis=1)

    # Augmented features rbf kernel
    width_x = width_rbf(X)
    K_x = rbf_dot(X, X, width_x)
    
    # site features linear kernel
    K_d = np.dot(D, D.T)

    # Centering matrix
    n = X.shape[0]
    H = np.identity(n) - (1/n)*np.dot(np.ones((1,n)).T,np.ones((1,n)))

    if labels == False:
        # unsupervised MIDA
        
        mat = multi_dot([K_x, multi_dot([-1.*H, K_d, H]) + mu*H, K_x])
        eigs, eigv = np.linalg.eig(mat)
        ind = eigs.argsort()[-h:][::-1]
        W = eigv[:, ind]
    else:
        # semi supervised MIDA

        # label information linear kernel
        K_y = np.dot(Y, Y.T)

        mat = multi_dot([K_x, multi_dot([-1.*H,K_d,H]) + mu*H + gamma_y*multi_dot([H, K_y, H]), K_x])
        eigs, eigv = np.linalg.eig(mat)
        ind = eigs.argsort()[-h:][::-1]
        W = eigv[:, ind]


    # projected features
    Z = np.dot(W.T, K_x).T

    return Z

def site_information_mat(data, num_subjects = 1035, num_domains = 20):

    Y = data[:,1].reshape((num_subjects, 1))
    domain_features = np.zeros((num_subjects, num_domains))

    for i in range(num_subjects):
        domain_features[i, int(Y[i])] = 1

    return domain_features

def normalise_features(X_orig):
    X = X_orig -  X_orig.mean(axis=0)
    return X