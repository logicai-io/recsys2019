import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from recsys.metric import calculate_mean_rec_err
from recsys.submission import group_clickouts
from recsys.transformers import RankFeatures
from recsys.utils import timer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(0)


if __name__ == '__main__':
    df_all = pd.read_csv("../../data/events_sorted_trans.csv")
    train_sessions = 10000

    with timer('filtering training observations'):
        if train_sessions:
            print("Before splitting shape", df_all.shape[0])
            sample_sessions = df_all.query("src == 'train'")["session_id"].sample(train_sessions)
            df = df_all[df_all["session_id"].isin(sample_sessions)].reset_index()
            print("After splitting shape", df.shape[0])
        else:
            df = df_all
    print("Training data shape", df.shape)

    # item_metadata = pd.read_csv("../../data/item_metadata.csv")
    # df = pd.merge(df, item_metadata, on="item_id", how="left")
    # df["properties"].fillna("", inplace=True)

    numerical_features = ['rank',
                          'price',
                          'clickout_user_item_clicks',
                          'clickout_item_clicks',
                          'clickout_item_impressions',
                          'was_interaction_img',
                          'interaction_img_freq',
                          'was_interaction_deal',
                          'interaction_deal_freq',
                          'was_interaction_info',
                          'interaction_info_freq']
    numerical_features_for_ranking = [
        'price',
        'clickout_user_item_clicks',
        'clickout_item_clicks',
        'clickout_item_impressions',
        'interaction_img_freq',
        'interaction_deal_freq',
        'interaction_info_freq'
    ]
    categorical_features = ["device", "platform"]  # , "current_filters"]

    with timer('splitting timebased'):
        split_timestamp = np.percentile(df.timestamp, 70)
        df_train = df[df["timestamp"] < split_timestamp]
        df_val = df[(df["timestamp"] > split_timestamp)]

    vectorizer = ColumnTransformer([
        ('numerical', make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), numerical_features),
        # ('feat_eng', make_pipeline(FeatureEng(), StandardScaler()), numerical_features),
        ('categorical', make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown='ignore')),
         categorical_features),
        ('numerical_ranking', RankFeatures(), numerical_features_for_ranking + ["clickout_id"])
        # ('properties',
        #  CountVectorizer(preprocessor=lambda x: "UNK" if x != x else x, tokenizer=lambda x: x.split("|"), min_df=5),
        #  "properties")
    ])

    with timer('fitting'):
        model = make_pipeline(vectorizer, LGBMClassifier(n_estimators=100, n_jobs=-2))
        model.fit(df_train, df_train["was_clicked"])

    train_pred = model.predict_proba(df_train)[:, 1]
    print(roc_auc_score(df_train["was_clicked"].values, train_pred))

    val_pred = model.predict_proba(df_val)[:, 1]
    print(roc_auc_score(df_val["was_clicked"].values, val_pred))

    df_val["click_proba"] = val_pred

    sessions_items, _ = group_clickouts(df_val)

    val_check = df_val[df_val["was_clicked"] == 1][["clickout_id", "item_id"]]
    val_check["predicted"] = val_check["clickout_id"].map(sessions_items)
    print("Validation MRE", calculate_mean_rec_err(val_check["predicted"].tolist(), val_check["item_id"]))

    df_test = df_all.query("is_test==1")
    df_test["click_proba"] = model.predict_proba(df_test)[:, 1]

    print("""For some reason it can be that click_proba is much greater 
    for test data. It is not a good idea to submit something if 
    the mean click_proba for test is much greater than for validation""")

    print(
        "Mean click proba validation {:2f} test {:2f}".format(df_val["click_proba"].mean(),
                                                              df_test["click_proba"].mean()))
    _, submission_df = group_clickouts(df_test)
    submission_df.to_csv("submission.csv", index=False)
