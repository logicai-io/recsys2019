import gc
import pathlib
import warnings

import click
import numpy as np
import pandas as pd
import pyarrow.feather as pf
from lightgbm import LGBMClassifier, LGBMRanker
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts
from recsys.utils import group_lengths, reduce_mem_usage, timer
from recsys.vectorizers import make_vectorizer_1, make_vectorizer_2
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
SORTED_TRANS_CSV = pathlib.Path().absolute().parents[1] / "data" / "events_sorted_trans.csv"
SORTED_TRANS_FEATHER = pathlib.Path().absolute().parents[1] / "data" / "events_sorted_trans.feather"

np.random.seed(0)


class Model:
    def __init__(self, vectorizer, model, weight, is_prob):
        self.vectorizer = vectorizer
        self.model = model
        self.weight = weight
        self.is_prob = is_prob

    def fit_and_predict(self, df_train, df_val, validate=False):
        mat_train = self.vectorizer.fit_transform(df_train)
        mat_val = self.vectorizer.transform(df_val)
        if isinstance(self.model, LGBMRanker):
            self.model.fit(
                mat_train, df_train["was_clicked"].values, group=group_lengths(df_train["clickout_id"].values)
            )
        else:
            self.model.fit(mat_train, df_train["was_clicked"].values)
        if self.is_prob:
            val_pred = self.model.predict_proba(mat_val)[:, 1]
            if validate:
                train_pred = self.model.predict_proba(mat_train)[:, 1]
                print(roc_auc_score(df_train["was_clicked"].values, train_pred))
                print(roc_auc_score(df_val["was_clicked"].values, val_pred))
        else:
            val_pred = self.model.predict(mat_val)
            if validate:
                train_pred = self.model.predict(mat_train)
                print(roc_auc_score(df_train["was_clicked"].values, train_pred))
                print(roc_auc_score(df_val["was_clicked"].values, val_pred))
        return val_pred


class ModelTrain:
    models = [
        Model(make_vectorizer_1(), LGBMClassifier(n_estimators=200, n_jobs=-2), weight=1.0, is_prob=True),
        Model(make_vectorizer_2(), LGBMRanker(n_estimators=200, n_jobs=-2), weight=0.2, is_prob=False),
    ]

    def __init__(self, n_jobs=-2, reduce_df_memory=False, load_feather=False):
        self.n_jobs = n_jobs
        self.reduce_df_memory = reduce_df_memory
        self.load_feather = load_feather

    def validate_models(self, n_users, n_debug=None):
        df_train, df_val = self.load_train_val_2(n_users, n_debug=n_debug)
        preds = np.vstack(
            [model.weight * model.fit_and_predict(df_train, df_val, validate=True) for model in self.models]
        ).sum(axis=0)
        df_val["click_proba"] = preds
        print("MRR {:4f}".format(mrr_fast(df_val, "click_proba")))

    def submit_models(self, n_users):
        df_train, df_test = self.load_train_test(n_users)
        preds = np.vstack([model.weight * model.fit_and_predict(df_train, df_test) for model in self.models]).sum(
            axis=0
        )
        df_test["click_proba"] = preds
        _, submission_df = group_clickouts(df_test)
        submission_df.to_csv("submission.csv", index=False)

    def load_train_val(self, n_users, n_debug=None, train_on_test_users=True):
        with timer("Reading training data"):
            if n_debug:
                df_all = pd.read_csv(SORTED_TRANS_CSV, nrows=n_debug)
            else:
                if self.load_feather:
                    df_all = pf.read_feather(SORTED_TRANS_FEATHER)
                else:
                    df_all = pd.read_csv(SORTED_TRANS_CSV)
                if self.reduce_df_memory:
                    df_all = reduce_mem_usage(df_all)
                if n_users:
                    train_users = set(
                        np.random.choice(df_all[df_all["is_test"] == 0].user_id.unique(), n_users, replace=False)
                    )
                    if train_on_test_users:
                        train_users |= set(df_all[df_all["is_test"] == 1].user_id.unique())
                    df_all = df_all[(df_all.user_id.isin(train_users)) & (df_all.is_test == 0)]
            print("Training on {} users".format(df_all["user_id"].nunique()))
            print("Training data shape", df_all.shape)
        with timer("splitting timebased"):
            split_timestamp = np.percentile(df_all.timestamp, 90)
            df_train = df_all[df_all["timestamp"] <= split_timestamp]
            df_val = df_all[(df_all["timestamp"] > split_timestamp)]
        return df_train, df_val

    def load_train_val_2(self, n_users, n_debug=None, train_on_test_users=True):
        with timer("Reading training data"):
            if n_debug:
                df_all = pd.read_csv(SORTED_TRANS_CSV, nrows=n_debug)
            else:
                if self.load_feather:
                    df_all = pf.read_feather(SORTED_TRANS_FEATHER)
                else:
                    df_all = pd.read_csv(SORTED_TRANS_CSV)
                if self.reduce_df_memory:
                    df_all = reduce_mem_usage(df_all)

                if n_users:
                    users_sample = set(df_all["user_id"].sample(n_users).unique())
                    df_all = df_all[df_all["user_id"].isin(users_sample)]
                    gc.collect()

            df_all["clickout_rank"] = df_all.groupby("user_id")["clickout_id"].rank("dense", ascending=False)
            split_timestamp = np.percentile(df_all.timestamp, 20)
            df_train = df_all[(df_all["clickout_rank"] > 1) | (df_all["timestamp"] <= split_timestamp)]
            df_val = df_all[(df_all["clickout_rank"] == 1) & (df_all["timestamp"] > split_timestamp)]

            print("Training on {} users".format(df_train["user_id"].nunique()))
            print("Validating on {} users".format(df_val["user_id"].nunique()))

        return df_train, df_val

    def load_train_test(self, n_users, train_on_test_users=True):
        with timer("Reading training and testing data"):
            if self.load_feather:
                df_all = pf.read_feather(SORTED_TRANS_FEATHER)
            else:
                df_all = pd.read_csv(SORTED_TRANS_CSV)
            if self.reduce_df_memory:
                df_all = reduce_mem_usage(df_all)
            df_test = df_all[df_all["is_test"] == 1]
            if n_users:
                train_users = set(
                    np.random.choice(df_all[df_all["is_test"] == 0].user_id.unique(), n_users, replace=False)
                )
                if train_on_test_users:
                    train_users |= set(df_all[df_all["is_test"] == 1].user_id.unique())
                df_all = df_all[(df_all.user_id.isin(train_users)) & (df_all.is_test == 0)]
            print("Training on {} users".format(df_all["user_id"].nunique()))
            print("Training data shape", df_all.shape)
        return df_all, df_test


@click.command()
@click.option("--n_users", type=int, default=None, help="Number of users to user for training")
@click.option("--n_jobs", type=int, default=-2, help="Number of cores to run models on")
@click.option("--n_debug", type=int, default=None, help="Number of rows to use for debuging")
@click.option("--action", type=str, default="validate", help="What to do: validate/submit")
@click.option("--reduce_df_memory", type=bool, default=True, help="Aggresively reduce DataFrame memory")
@click.option("--load_feather", type=bool, default=False, help="Use .feather or .csv DataFrame")
def main(n_users, n_jobs, n_debug, action, reduce_df_memory, load_feather):
    print(f"n_users={n_users}")
    print(f"action={action}")
    print(f"n_jobs={n_jobs}")
    print(f"reduce_df_memory={reduce_df_memory}")
    print(f"load_feather={load_feather}")
    trainer = ModelTrain(n_jobs=n_jobs, reduce_df_memory=reduce_df_memory, load_feather=load_feather)
    if action == "validate":
        with timer("validating models"):
            trainer.validate_models(n_users, n_debug)
    elif action == "submit":
        with timer("training full data models"):
            trainer.submit_models(n_users)


if __name__ == "__main__":
    main()
