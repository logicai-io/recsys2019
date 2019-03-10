import json
import warnings

import click
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from recsys.metric import calculate_mean_rec_err
from recsys.submission import group_clickouts
from recsys.utils import group_lengths, timer
from recsys.vectorizers import make_vectorizer_1, make_vectorizer_2, numerical_features
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

np.random.seed(0)


def train_models(nrows):
    df_all = pd.read_csv("../../data/events_sorted_trans.csv", nrows=nrows).query("src == 'train'")
    df_all["last_event_ts"] = df_all["last_event_ts"].map(json.loads)
    print(df_all["session_id"].nunique())

    train_sessions = None

    with timer("filtering training observations"):
        if train_sessions:
            print("Before splitting shape", df_all.shape[0])
            sample_users = df_all.query("src == 'train'")["user_id"].sample(
                train_sessions
            )
            df = df_all[df_all["user_id"].isin(sample_users)].reset_index()
            print("After splitting shape", df.shape[0])
        else:
            df = df_all
    print("Training data shape", df.shape)

    print("Correlations of numerical features")
    for col in numerical_features:
        if col in df.columns:
            print(col, pearsonr(df[col], df["was_clicked"]))

    with timer("splitting timebased"):
        split_timestamp = np.percentile(df.timestamp, 70)

    df_train = df[df["timestamp"] < split_timestamp]
    df_val = df[(df["timestamp"] > split_timestamp)]

    with timer("lgb"):
        vectorizer_1 = make_vectorizer_1()
        clf = LGBMClassifier(n_estimators=200, n_jobs=-2)
        model_lgb = make_pipeline(vectorizer_1, clf)
        model_lgb.fit(df_train, df_train["was_clicked"])

        train_pred_lgb = model_lgb.predict_proba(df_train)[:, 1]
        print(roc_auc_score(df_train["was_clicked"].values, train_pred_lgb))

        val_pred_lgb = model_lgb.predict_proba(df_val)[:, 1]
        print(roc_auc_score(df_val["was_clicked"].values, val_pred_lgb))

    with timer("lgb rank"):
        vectorizer_2 = make_vectorizer_2()
        ranker = LGBMRanker(n_estimators=200, n_jobs=-2)
        model_lgbrank = make_pipeline(vectorizer_2, ranker)
        model_lgbrank.fit(
            df_train,
            df_train["was_clicked"].values,
            lgbmranker__group=group_lengths(df_train["clickout_id"].values),
        )
        train_pred_lgbrank = model_lgbrank.predict(df_train)
        print(roc_auc_score(df_train["was_clicked"].values, train_pred_lgbrank))
        val_pred_lgbrank = model_lgbrank.predict(df_val)
        print(roc_auc_score(df_val["was_clicked"].values, val_pred_lgbrank))

    df_val["click_proba"] = val_pred_lgb + val_pred_lgbrank * 0.2
    sessions_items, _ = group_clickouts(df_val)
    val_check = df_val[df_val["was_clicked"] == 1][["clickout_id", "item_id"]]
    val_check["predicted"] = val_check["clickout_id"].map(sessions_items)
    print(
        "Validation MRE",
        calculate_mean_rec_err(val_check["predicted"].tolist(), val_check["item_id"]),
    )

    return [(1.0, model_lgb), (0.2, model_lgbrank)]


def read_test():
    df_all = pd.read_csv("../../data/events_sorted_trans.csv")
    df_all["last_event_ts"] = df_all["last_event_ts"].map(json.loads)
    return df_all.query("is_test==1")


def make_test_predictions(models):
    df_test = read_test()
    df_test["click_proba"] = (
            models[0][1].predict_proba(df_test)[:, 1] + models[1][1].predict(df_test) * 0.2
    )
    _, submission_df = group_clickouts(df_test)
    submission_df.to_csv("submission.csv", index=False)


@click.command()
@click.option("--limit", type=int, default=None, help="Number of rows to process")
@click.option("--submit", type=bool, default=False, help="Prepare submission file")
def main(limit, submit):
    with timer("training models"):
        models = train_models(limit)
    if submit:
        with timer("predicting"):
            make_test_predictions(models)


if __name__ == "__main__":
    main()
