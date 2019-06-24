import json

import pandas as pd
from lightgbm import LGBMRanker, LGBMRankerMRR2, LGBMRankerMRR3, LGBMRankerMRR4
from recsys.data_generator.accumulators import (AccByKey, ActionsTracker, ItemCTR, ItemCTRInteractions, PropertiesBayes)
from recsys.data_generator.generate_training_data import FeatureGenerator
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast, mrr_fast_v2
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_3_no_eng
from warnings import filterwarnings

filterwarnings('ignore')

accumulators = [
    ActionsTracker(),
    ItemCTR(action_types=["clickout item"]),
    AccByKey(ItemCTR(action_types=["clickout item"]), key="platform_device"),
    AccByKey(ItemCTR(action_types=["clickout item"]), key="platform"),
    AccByKey(ItemCTR(action_types=["clickout item"]), key="device"),
    ItemCTRInteractions(),
    AccByKey(ItemCTRInteractions(), key="platform_device"),
    AccByKey(ItemCTRInteractions(), key="platform"),
    AccByKey(ItemCTRInteractions(), key="device"),
    PropertiesBayes(path="../../data/item_metadata_map.joblib"),
    # AccByKey(PropertiesBayes(path="../../data/item_metadata_map.joblib"), key="platform"),
]

csv = "../../data/events_sorted_trans_mini.csv"
feature_generator = FeatureGenerator(
    limit=100000,
    accumulators=accumulators,
    save_only_features=False,
    input="../../data/events_sorted.csv",
    save_as=csv,
)
feature_generator.generate_features()

df = pd.read_csv(csv)
df["actions_tracker"] = df["actions_tracker"].map(json.loads)

features = [col for col in list(df.columns[27:]) if col != "actions_tracker"]
df_train, df_val = split_by_timestamp(df)
vectorizer = make_vectorizer_3_no_eng(numerical_features=features, numerical_features_for_ranking=features)

mat_train = vectorizer.fit_transform(df_train, df_train["was_clicked"])
print(mat_train.shape)
mat_val = vectorizer.transform(df_val)
print(mat_val.shape)

def mrr_metric(train_data, preds):
    mrr = mrr_fast_v2(train_data, preds, df_val["clickout_id"].values)
    return "error", mrr, True

for est in [LGBMRanker, LGBMRankerMRR2, LGBMRankerMRR3, LGBMRankerMRR4]:
    params = {"learning_rate":0.1, "n_estimators":900, "min_child_samples":5, "min_child_weight":0.00001, "n_jobs":-2}
    model = est(**params)
    print(model)
    model.fit(
        mat_train,
        df_train["was_clicked"],
        group=group_lengths(df_train["clickout_id"]),
    )

    df_train["click_proba"] = model.predict(mat_train)
    df_val["click_proba"] = model.predict(mat_val)

    print("Train", mrr_fast(df_train, "click_proba"))
    print("Val", mrr_fast(df_val, "click_proba"))
    print("By rank")
    for n in range(1, 10):
        print(n, mrr_fast(df_val[df_val["clickout_step_rev"] == n], "click_proba"))
    print()


"""
lambdarank
Train 0.8501484161745311
Val 0.5861931622195132
By rank
1 0.6338783791012323
2 0.5662563029336679
3 0.5251697068363734
4 0.5566058514135438
5 0.6073567014702749
6 0.4824402224657979
7 0.42239138738499354
8 0.32799584674584675
9 0.3165779114676174

LGBMRankerMRR2(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        importance_type='split', learning_rate=0.1, max_depth=-1,
        min_child_samples=5, min_child_weight=1e-05, min_split_gain=0.0,
        n_estimators=900, n_jobs=-2, num_leaves=31, objective=None,
        random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
lambdarank_mrr2
Train 0.7635372891218932
Val 0.5939604100331864
By rank
1 0.6341420966210083
2 0.5772701212932596
3 0.5458590623742139
4 0.5628646951339259
5 0.6979055666555667
6 0.4732547645591123
7 0.4380561672084164
8 0.40625862920193695
9 0.42953408306669183

LGBMRankerMRR3(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        importance_type='split', learning_rate=0.1, max_depth=-1,
        min_child_samples=5, min_child_weight=1e-05, min_split_gain=0.0,
        n_estimators=900, n_jobs=-2, num_leaves=31, objective=None,
        random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
lambdarank_mrr3
Train 0.8515840828365762
Val 0.5787894899696383
By rank
1 0.6236870271967243
2 0.5645266588461961
3 0.5049953406062451
4 0.5582546636668995
5 0.6162863559728883
6 0.4394555927164622
7 0.4035535782940627
8 0.48395061728395067
9 0.3974354056003672

"""