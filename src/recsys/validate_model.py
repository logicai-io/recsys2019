import json
import warnings

import click
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts
from recsys.utils import group_lengths, reduce_mem_usage, timer
from recsys.vectorizers import make_vectorizer_1, make_vectorizer_2
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

np.random.seed(0)


class ModelTrain():

    def __init__(self, n_jobs=-2, reduce_df_memory=False):
        self.n_jobs = n_jobs
        self.reduce_df_memory = reduce_df_memory

    def train_models(self, n_rows):
        df_train, df_val = self.load_train_val(n_rows)
        model_lgb, val_pred_lgb = self.fit_lgbm(df_train, df_val)
        model_lgbrank, val_pred_lgbrank = self.fit_lgbm_rank(df_train, df_val)
        df_val["click_proba"] = val_pred_lgb + val_pred_lgbrank * 0.2
        print("Validation MRR {:.4f}".format(mrr_fast(df_val, "click_proba")))
        return [(1.0, model_lgb), (0.2, model_lgbrank)]

    def fit_lgbm_rank(self, df_train, df_val):
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
        return model_lgbrank, val_pred_lgbrank

    def fit_lgbm(self, df_train, df_val):
        with timer("lgb"):
            vectorizer_1 = make_vectorizer_1()
            clf = LGBMClassifier(n_estimators=200, n_jobs=-2)
            model_lgb = make_pipeline(vectorizer_1, clf)
            model_lgb.fit(df_train, df_train["was_clicked"])
            train_pred_lgb = model_lgb.predict_proba(df_train)[:, 1]
            print(roc_auc_score(df_train["was_clicked"].values, train_pred_lgb))
            val_pred_lgb = model_lgb.predict_proba(df_val)[:, 1]
            print(roc_auc_score(df_val["was_clicked"].values, val_pred_lgb))
        return model_lgb, val_pred_lgb

    def load_train_val(self, nrows):
        df_all = pd.read_csv("../../data/events_sorted_trans.csv", nrows=nrows).query(
            "src == 'train'"
        )
        if self.reduce_df_memory:
            df_all = reduce_mem_usage(df_all)
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
        df["clicked_before"] = (df["item_id"] == df["last_item_clickout"]).astype(np.int)
        print("Training data shape", df.shape)
        with timer("splitting timebased"):
            split_timestamp = np.percentile(df.timestamp, 70)
            df_train = df[df["timestamp"] < split_timestamp]
            df_val = df[(df["timestamp"] > split_timestamp)]
        return df_train, df_val

    def read_test(self):
        df_all = pd.read_csv("../../data/events_sorted_trans.csv")
        if self.reduce_df_memory:
            df_all = reduce_mem_usage(df_all)
        df_all["last_event_ts"] = df_all["last_event_ts"].map(json.loads)
        return df_all.query("is_test==1")

    def make_test_predictions(self, models):
        df_test = self.read_test()
        df_test["click_proba"] = (
                models[0][1].predict_proba(df_test)[:, 1] + models[1][1].predict(df_test) * 0.2
        )
        _, submission_df = group_clickouts(df_test)
        submission_df.to_csv("submission.csv", index=False)


@click.command()
@click.option("--limit", type=int, default=None, help="Number of rows to process")
@click.option("--n_jobs", type=int, default=-2, help="Number of cores to run models on")
@click.option("--submit", type=bool, default=True, help="Prepare submission file")
@click.option("--reduce_df_memory", type=bool, default=True, help="Aggresively reduce DataFrame memory")
def main(limit, submit, n_jobs, reduce_df_memory):
    print(f"submit={submit}")
    print(f"n_jobs={n_jobs}")
    print(f"reduce_df_memory={reduce_df_memory}")
    trainer = ModelTrain(n_jobs=n_jobs, reduce_df_memory=reduce_df_memory)
    with timer("training models"):
        models = trainer.train_models(limit)
    if submit:
        with timer("predicting"):
            trainer.make_test_predictions(models)


if __name__ == "__main__":
    main()
