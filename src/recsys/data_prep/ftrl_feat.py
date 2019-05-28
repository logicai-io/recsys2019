"""
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
"""
import pathlib
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt

import joblib
from tqdm import tqdm

PATH_TO_IMM = "../../../data/item_metadata_map.joblib"
# imm = joblib.load(PATH_TO_IMM)

# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
train = "../../../data/events_sorted_trans.csv"  # path to training file

# B, model
alpha = 0.1  # learning rate
beta = 1.0  # smoothing parameter for adaptive learning rate
L1 = 1.0  # L1 regularization, larger value means more regularized
L2 = 1.0  # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 30  # number of weights to use
interaction = True  # whether to enable poly2 feature interactions

##############################################################################
# class, function, generator definitions #####################################
##############################################################################


class ftrl_proximal(object):
    """ Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    """

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.0] * D
        self.z = [0.0] * D
        self.w = {}

    def _indices(self, x):
        """ A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        """

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in range(L):
                for j in range(i + 1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + "_" + str(x[j]))) % D

    def predict(self, x):
        """ Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        """

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.0
        for i in self._indices(x):
            sign = -1.0 if z[i] < 0 else 1.0  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.0
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1.0 / (1.0 + exp(-max(min(wTx, 35.0), -35.0)))

    def update(self, x, p, y):
        """ Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        """

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    """ FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    """

    p = max(min(p, 1.0 - 10e-15), 10e-15)
    return -log(p) if y == 1.0 else -log(1.0 - p)


def data(path, D):
    """ GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    """

    for t, row in enumerate(DictReader(open(path))):
        # process id
        # process clicks
        y = 0.0 if row["was_clicked"] == "0" else 1.0

        # build x
        x = []
        for key in [
            "rank",
            "interaction_img_freq",
            "interaction_deal_freq",
            "interaction_info_freq",
            "interaction_rating_freq",
            "interaction_img_diff_ts",
            "last_index_1",
            "last_index_2",
            "price",
            "user_id",
            "item_id"
            # "platform",
            # "device",
        ]:

            value = row[key]

            # one-hot encode everything with hash trick
            index = abs(hash(key + "_" + value)) % D
            x.append(index)

        yield t, row, x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

# start training
loss = 0.0
count = 0

predictions = []

for t, row, x, y in tqdm(data(train, D)):  # data is a generator
    p = learner.predict(x)
    predictions.append(p)
    loss += logloss(p, y)
    count += 1
    if t % 10000 == 0:
        print(loss / count)
    if row["is_test"] == "0":
        learner.update(x, p, y)

print("Epoch %d finished, validation logloss: %f, elapsed time: %s" % (1, loss / count, str(datetime.now() - start)))

##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################

with open("ftrl_prob.csv", "w") as outfile:
    outfile.write("click\n")
    for p in predictions:
        outfile.write("%s\n" % (str(p)))
