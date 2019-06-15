import pandas as pd
from recsys.data_generator.accumulators import (
    GlobalClickoutTimestamp,
    PairwiseCTR,
    RankBasedCTR,
    RankOfItemsFreshClickout,
    SequenceClickout,
)
from recsys.data_generator.generate_training_data import FeatureGenerator
from recsys.utils import get_sort_index
from recsys.vectorizers import make_vectorizer_4
from scipy import sparse as sp
from scipy.sparse import save_npz

csv = "../../data/events_sorted_trans_v2.csv"
df = pd.read_csv(csv)
df["sort_index"] = df.apply(get_sort_index, axis=1)
df.sort_values("sort_index", inplace=True)
df.drop("sort_index", axis=1)

features = list(df.columns)[27:]

df_train = df[(df["is_val"] == 0) & (df["is_test"] == 0)]
vectorizer = make_vectorizer_4(numerical_features=features, numerical_features_for_ranking=features)
vectorizer.fit(df_train, df_train["was_clicked"])

mat = pd.DataFrame(vectorizer.transform(df))
mat.to_hdf("../../data/proc/vectorizer_2/data_v1_Xdense.npz", key="data", mode="w")

df[
    [
        "user_id",
        "session_id",
        "platform",
        "device",
        "city",
        "timestamp",
        "step",
        "clickout_id",
        "item_id",
        "src",
        "is_test",
        "is_val",
        "was_clicked",
    ]
].to_hdf("../../data/proc/vectorizer_2/data_v1_meta.h5", key="data", mode="w")
