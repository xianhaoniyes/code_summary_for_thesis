
from emd import emd
import numpy as np
import time


# Compute the EMD kernel
# X1 and X2 are lists with data from multiple participants as elements
def ComputeKernelEMD(X1 ,X2, dist):
    sz1 = np.shape(X1)[0]
    sz2 = np.shape(X2)[0]
    D = np.zeros((sz1, sz2))
    for i in range(0, sz1):
        for j in range(i, sz2):

            D[i, j] = (emd(X1[i], X2[j], distance=dist))
    D = D + np.transpose(np.triu(D, k=1))
    return D

def ComputeKernelEMD1D(X1,X2,dist):
    sz1 = np.shape(X1)[0]
    sz2 = np.shape(X2)[0]
    D = np.zeros((sz1, sz2))
    for i in range(0, sz1):
        for j in range(i, sz2):
            D[i, j] = (emd(X1[i], X2[j], distance=dist))

    D = np.squeeze(D)
    return D


def AdvancedKernelEMD(X1, X1_labels, X2, X2_labels, dist):
    sz1 = np.shape(X1)[0]
    sz2 = np.shape(X2)[0]
    D = np.zeros((sz1, sz2))
    for i in range(0, sz1):
        for j in range(i, sz2):
            s1 = X1[i]
            X1_positive = s1[X1_labels[i] == 1]
            X1_negative = s1[X1_labels[i] == 0]

            s2 = X2[j]
            X2_positive = s2[X2_labels[j] == 1]
            X2_negative = s2[X2_labels[j] == 0]
            D[i, j] = (emd(X1_positive, X2_positive, distance=dist))\
                      +(emd(X1_negative, X2_negative, distance=dist))


    D = D + np.transpose(np.triu(D, k=1))
    return D

def NegativeKernelEMD(X1, X1_labels, X2, X2_labels, dist):
    sz1 = np.shape(X1)[0]
    sz2 = np.shape(X2)[0]
    D = np.zeros((sz1, sz2))
    for i in range(0, sz1):
        for j in range(i, sz2):
            s1 = X1[i]

            X1_negative = s1[X1_labels[i] == 0]

            s2 = X2[j]

            X2_negative = s2[X2_labels[j] == 0]
            D[i, j] = (emd(X1_negative, X2_negative, distance=dist))
    D = D + np.transpose(np.triu(D, k=1))

    return D



def ComputeKernelEMD_weights(X1 ,X2,x_weights, y_weights, dist):
    sz1 = np.shape(X1)[0]
    sz2 = np.shape(X2)[0]
    D = np.zeros((sz1, sz2))
    for i in range(0, sz1):
        for j in range(i, sz2):
            D[i, j] = (emd(X1[i], X2[j], X_weights=x_weights[i],Y_weights=y_weights[j], distance=dist))
    D = D + np.transpose(np.triu(D, k=1))
    return D