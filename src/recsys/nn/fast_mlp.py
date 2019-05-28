import logging

import numpy as np
from mercari.fast_mlp_c import fit_epoch, fit_epoch_2_layers
from scipy import sparse as sp
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger("mercari")


def logistic(x):
    return 1 / (1 + np.exp(-x))


def softplus(x):
    return np.log(1 + np.exp(x))


def relu(x):
    return np.maximum(x, 0)


class FastMLPRegressor(BaseEstimator, RegressorMixin):
    ACTFUNC_MAP = {
        "logistic": (logistic, 3),
        "tanh": (np.tanh, 2),
        "relu": (relu, 1)
    }

    LOSS_MAP = {
        "squared": 1,
        "huber": 2
    }

    def __init__(self, nhidden1=50, nhidden2=None, nepochs=10, lambd=0.0, eta=0.05,
                 etadecay=0.25, seed=1, sigma=0.05, dropout=0.0, actfunc="logistic",
                 loss="squared", batchsize=1, evalfunc=None, verbose=True,
                 keep_epochs=False, huber_sigma=1.0):
        self.nhidden1 = nhidden1
        self.nhidden2 = nhidden2
        self.nepochs = nepochs
        self.lambd = lambd
        self.eta = eta
        self.etadecay = etadecay
        self.actfunc = actfunc
        self.loss = loss
        self.seed = seed
        self.sigma = sigma
        self.dropout = dropout
        self.batchsize = batchsize
        self.evalfunc = evalfunc
        self.verbose = verbose
        self.keep_epochs = keep_epochs
        self.huber_sigma = huber_sigma
        if self.keep_epochs:
            self.W1_history = []
            self.W2_history = []
            self.V_history = []
        self.rs = np.random.RandomState(self.seed)

    def fit(self, X, y, init=True):
        X = sp.csr_matrix(X).astype(np.float32)
        Y = y.astype(np.float32)

        if init:
            self.W1 = self.init_weights(X.shape[1], self.nhidden1)
            if self.nhidden2:
                self.W2 = self.init_weights(self.nhidden1, self.nhidden2)
                self.V = self.init_weights(self.nhidden2, 1)
            else:
                self.V = self.init_weights(self.nhidden1, 1)

        for epoch in range(self.nepochs):
            if self.verbose: print("epoch {}...".format(epoch))
            eta = self.eta / (epoch + 1) ** self.etadecay
            if self.batchsize == 1:
                if self.nhidden2:
                    fit_epoch_2_layers(X, Y, self.W1, self.W2, self.V, self.ACTFUNC_MAP[self.actfunc][1], eta,
                                       self.dropout, self.lambd,  self.LOSS_MAP[self.loss], self.huber_sigma)
                else:
                    fit_epoch(X, Y, self.W1, self.V, self.ACTFUNC_MAP[self.actfunc][1], eta,
                              self.dropout, self.lambd, self.LOSS_MAP[self.loss], self.huber_sigma)
            else:
                assert NotImplemented()
            if self.evalfunc != None and epoch % 5 == 4: self.evalfunc(self)

            if self.keep_epochs:
                self.W1_history.append(self.W1.copy())
                if self.nhidden2:
                    self.W2_history.append(self.W2.copy())
                self.V_history.append(self.V.copy())
        return self

    def init_weights(self, n_in, n_out):
        const = 2.0 if self.actfunc == 'logistic' else 6.0
        variance = const / (n_in + n_out)
        stddev = np.sqrt(variance)
        return self.rs.normal(0, stddev, (n_in, n_out)).astype(np.float32)

    def predict(self, X):
        if self.keep_epochs:
            return self.predict_all_epochs(X)
        else:
            return self.predict_last_epoch(X)

    def predict_last_epoch(self, X):
        X = sp.csr_matrix(X).astype(np.float32)
        act = self.ACTFUNC_MAP[self.actfunc][0]
        if self.nhidden2:
            W1 = self.W1
            W2 = self.W2
            V = self.V  # * (1 - self.params.dropout)
            return act(act(X.dot(W1)).dot(W2)).dot(V).reshape(-1)
        else:
            W1 = self.W1
            V = self.V  # * (1 - self.params.dropout)
            return act(X.dot(W1)).dot(V).reshape(-1)

    def predict_all_epochs(self, X):
        X = sp.csr_matrix(X).astype(np.float32)
        return np.hstack([
            self.ACTFUNC_MAP[self.actfunc][0](X.dot(W)).dot(V)
            for W, V in zip(self.W_history, self.V_history)])
