import pandas as pd
from recsys.data_generator.accumulators import (GlobalClickoutTimestamp, PairwiseCTR, RankBasedCTR,
                                                RankOfItemsFreshClickout, SequenceClickout)
from recsys.data_generator.generate_training_data import FeatureGenerator
from recsys.vectorizers import make_vectorizer_3_no_eng, make_vectorizer_4
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
    accumulators=accumulators,
    save_only_features=False,
    input="../../data/events_sorted.csv",
    save_as=csv,
)
feature_generator.generate_features()
df = pd.read_csv(csv)
features = list(df.columns)[27:]

df_train = df[(df["is_val"] == 0) & (df["is_test"] == 0)]
df_val = df[(df["is_val"] == 1)]
df_test = df[(df["is_test"] == 0)]

vectorizer = make_vectorizer_4(numerical_features=features, numerical_features_for_ranking=features)

mat_train = vectorizer.fit_transform(df_train, df_train["was_clicked"])
mat_val = vectorizer.transform(df_val)
mat_test = vectorizer.transform(df_test)

mat = pd.concat([mat_train, mat_val, mat_test], axis=0)

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
save_npz("../../data_proc/vectorizer_2/data_v1_Xcsr.h2", mat)