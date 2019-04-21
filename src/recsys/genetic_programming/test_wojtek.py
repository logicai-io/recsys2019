# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
import pandas as pd
from lightgbm import LGBMRanker
from recsys.metric import mrr_fast
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
import numpy as np
import warnings
import joblib

warnings.filterwarnings("ignore")

df = pd.read_csv("../../../data/events_sorted_trans.csv")


def load_joblib_item_dict(path):
    d = joblib.load(path)
    d = {int(k): v for k, v in d.items()}
    return d


pagerank = load_joblib_item_dict("../../../data/pagerank_dict.joblib")
cluster = load_joblib_item_dict("../../../data/cluster_dict.joblib")
neigh = load_joblib_item_dict("../../../data/avg_neighbor_deg_dict.joblib")
cluster_triangle = load_joblib_item_dict("../../../data/cluster_triangles_dict.joblib")

for d, name in [(pagerank, "pagerank"), (cluster, "cluster"), (neigh, "neigh"), (cluster_triangle, "cluster_triangle")]:
    df["pagerank"] = df["item_id"].map(d)
    print(name, mrr_fast(df, "pagerank"))
    df["pagerank_rank"] = -df.groupby("clickout_id")["pagerank"].rank("max", ascending=False)
    print(name, mrr_fast(df, "pagerank_rank"))
    df["pagerank_rank"] = df.groupby("clickout_id")["pagerank"].rank("max", ascending=False)
    print(name, mrr_fast(df, "pagerank_rank"))
