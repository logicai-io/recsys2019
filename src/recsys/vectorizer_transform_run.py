import gc
import glob
import os

import h5sparse
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from recsys.transformers import (
    FeatureEng,
    FeaturesAtAbsoluteRank,
    LagNumericalFeaturesWithinGroup,
    MinimizeNNZ,
    PandasToNpArray,
    PandasToRecords,
    RankFeatures,
    SanitizeSparseMatrix,
    SparsityFilter,
    DivideByRanking,
)
from recsys.utils import logger
from recsys.vectorizers import make_vectorizer_1
from scipy.sparse import load_npz, save_npz
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    input_files = "../../data/proc/raw_csv/*.csv"
    vectorizer = make_vectorizer_1()
    df = pd.read_csv(sorted(glob.glob(input_files))[-1])
    vectorizer.fit(df)
    joblib.dump(vectorizer, "../../data/proc/vectorizer_1/vectorizer.joblib")

    filenames = Parallel(n_jobs=self.n_jobs)(
        delayed(self.vectorize_one)(fn) for fn in sorted(glob.glob(self.input_files))
    )
    metadata_fns, csr_fns = list(zip(*filenames))
    self.save_to_one_file_metadata(metadata_fns)
    self.save_to_one_flie_csrs(csr_fns)

    poll = p.poll()
    if poll == None:
