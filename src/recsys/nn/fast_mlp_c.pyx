# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport tanh, exp, sqrt

ctypedef np.int32_t DINT
ctypedef np.float32_t DFLOAT

cdef struct cs_t:
    int nzmax       # maximum number of entries
    int nrow          # number of rows
    int ncol           # number of columns
    int *indptr         # column pointers (size n+1) or col indices (size nzmax)
    int *indices          # row indices, size nzmax
    float *data       # numerical values, size nzmax

cdef cs_t _parsecsr(mat):
    cdef int i
    cdef cs_t csX
    cdef np.ndarray[int, ndim=1, mode = 'c'] indptr  = mat.indptr
    cdef np.ndarray[int, ndim=1, mode = 'c'] indices = mat.indices
    cdef np.ndarray[DFLOAT, ndim=1, mode = 'c'] data = mat.data

    csX.nzmax = mat.data.shape[0]
    csX.nrow = mat.shape[0]
    csX.ncol = mat.shape[1]
    csX.indptr = &indptr[0]
    csX.indices = &indices[0]
    csX.data = &data[0]

    return csX

cdef int dot_sparse_vec(cs_t X,
                        int i,
                        DFLOAT[:,:] W,
                        int noutput,
                        DFLOAT[:] h,
                        DINT[:] hmask) nogil:
    cdef int j, k, j1, j2, jj;
    j1 = X.indptr[i]
    j2 = X.indptr[i + 1]
    for k in range(noutput):
        if (hmask[k]):
            continue
        h[k] = 0
        for jj in range(j1, j2):
            j = X.indices[jj]
            h[k] += X.data[jj] * W[j,k]
    return 1

cdef DFLOAT dot_vec_scalar(DFLOAT[:] h,
                           DFLOAT[:,:] V,
                           int ninput) nogil:
    cdef int l, k
    cdef DFLOAT yhat = 0
    for k in range(ninput):
        yhat += h[k]*V[k,0]
    return yhat


cdef DFLOAT dot_vec_vec(DFLOAT[:] h1,
                        DFLOAT[:] h2,
                        DFLOAT[:,:] W,
                        int ninput,
                        int noutput) nogil:
    cdef int i, k
    for k in range(noutput):
        h2[k] = 0
        for i in range(ninput):
            h2[k] += h1[i]*W[i,k]
    return 1

cdef DFLOAT logistic(DFLOAT x) nogil:
    return 1.0 / (1 + exp(-x))

cdef DFLOAT relu(DFLOAT x) nogil:
    if x < 0:
        return 0
    else:
        return x

cdef DFLOAT dtanh(DFLOAT hk) nogil:
    return 1 - hk*hk

cdef DFLOAT dlogistic(DFLOAT hk) nogil:
    return hk*(1-hk)

cdef DFLOAT drelu(DFLOAT hk) nogil:
    if hk < 0:
        return 0
    else:
        return 1


"""
https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
https://github.com/cgarciae/cybrain
"""

"""
huber loss derivative:

x / sqrt(x^2 / a^2 + 1)
"""

def fit_epoch(X_raw, DFLOAT[:] y, DFLOAT[:,:] W, DFLOAT[:,:] V, int act, float eta, float dropout, float lamb,
              int loss, float huber_sigma=1.0):
    cdef int nhidden = W.shape[1]
    cdef cs_t X = _parsecsr(X_raw)
    cdef DFLOAT[:] h = np.zeros(nhidden, dtype=np.float32)
    cdef DFLOAT yhat
    cdef DFLOAT[:] delta = np.zeros(nhidden, dtype=np.float32)
    cdef DINT[:] hmask = np.zeros(nhidden, dtype=np.int32)
    cdef int i, j, k, j1, j2, jj
    cdef float yl
    cdef DFLOAT gradl, gradk, a
    for i in range(X.nrow):
        dot_sparse_vec(X,i,W,nhidden,h,hmask)
        for k in range(nhidden):
            if act == 1: h[k] = relu(h[k])
            elif act == 2: h[k] = tanh(h[k])
            elif act == 3: h[k] = logistic(h[k])
        yhat = dot_vec_scalar(h,V,nhidden)

        a = (yhat - y[i])
        if loss == 1:
            gradl = 2*a  # squared_loss
        elif loss == 2:
            gradl = a / sqrt((a**2 / huber_sigma**2) + 1) # huber loss

        for k in range(nhidden):
            if hmask[k]:
                continue
            delta[k] = gradl * V[k,0]
            V[k,0] -= eta * (gradl*h[k] + lamb*V[k,0])

        j1 = X.indptr[i]
        j2 = X.indptr[i+1]
        for k in range(nhidden):
            if hmask[k]:
                continue
            if act == 1: gradk = delta[k] * drelu(h[k])
            elif act == 2: gradk = delta[k] * dtanh(h[k])
            elif act == 3: gradk = delta[k] * dlogistic(h[k])
            for jj in range(j1, j2):
                j = X.indices[jj]
                W[j,k] -= eta*(gradk * X.data[jj] + lamb*W[j,k])


def fit_epoch_parallel(X_raw, DFLOAT[:] y, DFLOAT[:,:] W, DFLOAT[:,:] V, int act, float eta, float dropout, float lamb,
              int loss, float huber_sigma=1.0):
    pass

"""
https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
https://github.com/cgarciae/cybrain
https://www.wolframalpha.com/input/?i=f(x)+%3D+(a%5E2)*(sqrt(1+%2B+(x%2Fa)%5E2)-1)
"""


def fit_epoch_2_layers(X_raw, DFLOAT[:] y, DFLOAT[:,:] W1, DFLOAT[:,:] W2, DFLOAT[:,:] V, int act, float eta,
                       float dropout, float lamb, int loss, float huber_sigma=1.0):
    cdef DFLOAT[:,:] dW1 = np.zeros_like(W1)
    cdef DFLOAT[:,:] dW2 = np.zeros_like(W2)
    cdef DFLOAT[:,:] dV = np.zeros_like(V)

    cdef int nhidden1 = W1.shape[1]
    cdef int nhidden2 = W2.shape[1]
    cdef cs_t X = _parsecsr(X_raw)
    cdef DFLOAT[:] h1 = np.zeros(nhidden1, dtype=np.float32)
    cdef DFLOAT[:] h2 = np.zeros(nhidden2, dtype=np.float32)
    cdef DFLOAT yhat
    cdef DFLOAT[:] delta1 = np.zeros(nhidden1, dtype=np.float32)
    cdef DFLOAT[:] delta2 = np.zeros(nhidden2, dtype=np.float32)
    cdef DINT[:] hmask1 = np.zeros(nhidden1, dtype=np.int32)
    cdef DINT[:] hmask2 = np.zeros(nhidden2, dtype=np.int32)
    cdef int i, j, k, j1, j2, jj
    cdef float yl
    cdef DFLOAT gradl, gradk1, gradk2
    for i in range(X.nrow):
        dot_sparse_vec(X,i,W1,nhidden1,h1,hmask1)
        for k in range(nhidden1):
            if act == 1: h1[k] = relu(h1[k])
            elif act == 2: h1[k] = tanh(h1[k])
            elif act == 3: h1[k] = logistic(h1[k])
        dot_vec_vec(h1, h2, W2, nhidden1, nhidden2)
        for k in range(nhidden2):
            if act == 1: h2[k] = relu(h2[k])
            elif act == 2: h2[k] = tanh(h2[k])
            elif act == 3: h2[k] = logistic(h2[k])
        yhat = dot_vec_scalar(h2,V,nhidden2)

        # backpropagate
        a = (yhat - y[i])
        if loss == 1:
            gradl = 2*a  # squared_loss
        elif loss == 2:
            gradl = a / sqrt((a**2 / huber_sigma**2) + 1)

        for k in range(nhidden2):
            if hmask2[k]:
                continue
            delta2[k] = gradl * V[k,0]
#             dV[k,0] = gradl*h2[k] + lamb*V[k,0]
            V[k,0] -= eta * (gradl*h2[k] + lamb*V[k,0])

        for k in range(nhidden1):
            delta1[k] = 0
        for k in range(nhidden2):
            if act == 1: gradk2 = delta2[k] * drelu(h2[k])
            elif act == 2: gradk2 = delta2[k] * dtanh(h2[k])
            elif act == 3: gradk2 = delta2[k] * dlogistic(h2[k])
            for j in range(nhidden1):
                delta1[j] += gradk2*W2[j,k]
#                 dW2[j,k] = gradk2 * h1[j] + lamb*W2[j,k]
                W2[j,k] -= eta*(gradk2 * h1[j] + lamb*W2[j,k])

        j1 = X.indptr[i]
        j2 = X.indptr[i+1]
        for k in range(nhidden1):
            if hmask1[k]:
                continue
            if act == 1: gradk1 = delta1[k] * drelu(h1[k])
            elif act == 2: gradk1 = delta1[k] * dtanh(h1[k])
            elif act == 3: gradk1 = delta1[k] * dlogistic(h1[k])
            for jj in range(j1, j2):
                j = X.indices[jj]
#                 dW1[j,k] = gradk1 * X.data[jj] + lamb*W1[j,k]
                W1[j,k] -= eta*(gradk1 * X.data[jj] + lamb*W1[j,k])

    return 1
