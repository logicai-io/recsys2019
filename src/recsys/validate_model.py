import pathlib
import warnings

import click
import numpy as np
import pandas as pd
import pyarrow.feather as pf
from lightgbm import LGBMClassifier, LGBMRanker
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts
from recsys.utils import group_lengths, reduce_mem_usage, timer
from recsys.vectorizers import make_vectorizer_1, make_vectorizer_2
from scipy.optimize import fmin, fmin_powell
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

    def normalize_predictions_by_group(self, pred, clickout_id):
        df = pd.DataFrame({"pred": pred, "clickout_id": clickout_id})
        df["pred_norm"] = df.groupby("clickout_id")["pred"].transform(lambda x: x / (x.std() + 1))
        return df["pred_norm"].values

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
                self.evaluate(df_train, df_val, train_pred, val_pred)
        else:
            val_pred = self.model.predict(mat_val)
            if validate:
                train_pred = self.model.predict(mat_train)
                self.evaluate(df_train, df_val, train_pred, val_pred)
        return val_pred

    def evaluate(self, df_train, df_val, train_pred, val_pred):
        print("Train AUC {:.4f}".format(roc_auc_score(df_train["was_clicked"].values, train_pred)))
        print("Val AUC {:.4f}".format(roc_auc_score(df_val["was_clicked"].values, val_pred)))
        df_val["click_proba"] = val_pred
        print("Val MRR {:.4f}".format(mrr_fast(df_val, "click_proba")))


class ModelTrain:
    def __init__(self, models, n_jobs=-2, reduce_df_memory=False, load_feather=False):
        self.models = models
        self.n_jobs = n_jobs
        self.reduce_df_memory = reduce_df_memory
        self.load_feather = load_feather

    def validate_models(self, n_users, n_debug=None):
        df_train, df_val = self.load_train_val(n_users, n_debug=n_debug)

        preds_mat = np.vstack([model.fit_and_predict(df_train, df_val, validate=True) for model in self.models]).T

        def opt_coefs(coefs):
            preds = preds_mat.dot(coefs)
            df_val["preds"] = preds
            mrr = mrr_fast(df_val, "preds")
            print(mrr, coefs)
            return -mrr

        best_coefs = fmin(opt_coefs, [model.weight for model in self.models])
        best_coefs = fmin_powell(opt_coefs, best_coefs)

        preds = preds_mat.dot(best_coefs)
        df_val["click_proba"] = preds
        print("MRR {:4f}".format(mrr_fast(df_val, "click_proba")))
        print("Best coefs: ", best_coefs)

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
            df_train, df_val = split_by_timestamp(df_all)
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


def info(action, load_feather, n_jobs, n_users, reduce_df_memory):
    print(f"n_users={n_users}")
    print(f"action={action}")
    print(f"n_jobs={n_jobs}")
    print(f"reduce_df_memory={reduce_df_memory}")
    print(f"load_feather={load_feather}")


@click.command()
@click.option("--n_users", type=int, default=None, help="Number of users to user for training")
@click.option("--n_trees", type=int, default=100, help="Number of trees for lightgbm models")
@click.option("--n_jobs", type=int, default=-2, help="Number of cores to run models on")
@click.option("--n_debug", type=int, default=None, help="Number of rows to use for debuging")
@click.option("--action", type=str, default="validate", help="What to do: validate/submit")
@click.option("--reduce_df_memory", type=bool, default=True, help="Aggresively reduce DataFrame memory")
@click.option("--load_feather", type=bool, default=False, help="Use .feather or .csv DataFrame")
def main(n_users, n_trees, n_jobs, n_debug, action, reduce_df_memory, load_feather):
    models = [
        Model(make_vectorizer_1(), LGBMClassifier(n_estimators=n_trees, n_jobs=n_jobs), weight=1.0, is_prob=True),
        Model(make_vectorizer_2(), LGBMRanker(n_estimators=n_trees, n_jobs=n_jobs), weight=0.2, is_prob=False),
    ]
    info(action, load_feather, n_jobs, n_users, reduce_df_memory)
    trainer = ModelTrain(models=models, n_jobs=n_jobs, reduce_df_memory=reduce_df_memory, load_feather=load_feather)
    if action == "validate":
        with timer("validating models"):
            trainer.validate_models(n_users, n_debug)
    elif action == "submit":
        with timer("training full data models"):
            trainer.submit_models(n_users)


if __name__ == "__main__":
    main()
