import pandas as pd
from recsys.data_generator.accumulators import (GlobalClickoutTimestamp, PairwiseCTR, RankBasedCTR,
                                                RankOfItemsFreshClickout, SequenceClickout)
from recsys.data_generator.generate_training_data import FeatureGenerator
from recsys.utils import get_sort_index
from recsys.vectorizers import make_vectorizer_4
from scipy import sparse as sp
from scipy.sparse import save_npz

accumulators = [
    PairwiseCTR(),
    RankOfItemsFreshClickout(),
    GlobalClickoutTimestamp(),
    SequenceClickout(),
    RankBasedCTR()
]

csv = "../../data/events_sorted_trans_v2.csv"
feature_generator = FeatureGenerator(
    limit=None,
    accumulators=accumulators,
    save_only_features=False,
    input="../../data/events_sorted.csv",
    save_as=csv,
)
feature_generator.generate_features()
