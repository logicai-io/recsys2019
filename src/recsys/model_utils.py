import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts
from recsys.utils import group_lengths, reduce_mem_usage, timer
from scipy.optimize import fmin, fmin_powell
from sklearn.metrics import roc_auc_score


class Model:
    def __init__(self, name, vectorizer, model, weight, is_prob):
        self.name = name
        self.vectorizer = vectorizer
        self.model = model
        self.weight = weight
        self.is_prob = is_prob

    def normalize_predictions_by_group(self, pred, clickout_id):
        df = pd.DataFrame({"pred": pred, "clickout_id": clickout_id})
        df["pred_norm"] = df.groupby("clickout_id")["pred"].transform(lambda x: x / (x.std() + 1))
        return df["pred_norm"].values

    def fit_and_predict(self, df_train, df_val, validate=False):
        with timer("vectorizing train"):
            mat_train = self.vectorizer.fit_transform(df_train)
            print("Train shape", mat_train.shape)
        with timer("vectorinzg val"):
            mat_val = self.vectorizer.transform(df_val)
            print("Val shape", mat_val.shape)

        with timer("fitting model"):
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
            print("Predicting validation")
            val_pred = self.model.predict(mat_val)
            if validate:
                print("Predicting train")
                train_pred = self.model.predict(mat_train)
                self.evaluate(df_train, df_val, train_pred, val_pred)
        self.save_predictions(df_val, val_pred, validate)
        return val_pred

    def save_predictions(self, df_val, val_pred, validate):
        df_preds = df_val[["user_id", "session_id", "timestamp", "step", "clickout_id", "item_id", "was_clicked"]]
        df_preds["pred"] = val_pred
        action = "validate" if validate else "test"
        df_preds.to_csv(f"predictions/{self.name}_{action}.csv", index=False)

    def evaluate(self, df_train, df_val, train_pred, val_pred):
        print("Train AUC {:.4f}".format(roc_auc_score(df_train["was_clicked"].values, train_pred)))
        print("Val AUC {:.4f}".format(roc_auc_score(df_val["was_clicked"].values, val_pred)))
        df_val["click_proba"] = val_pred
        print("Val MRR {:.4f}".format(mrr_fast(df_val, "click_proba")))


class ModelTrain:
    def __init__(self, models, datapath, n_jobs=-2, reduce_df_memory=False, load_feather=False):
        self.models = models
        self.datapath = datapath
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

    def load_train_val(self, n_users, n_debug=None):
        with timer("Reading training data"):
            if n_debug:
                df_all = pd.read_csv(self.datapath, nrows=n_debug)
            else:
                df_all = pd.read_csv(self.datapath)
                if self.reduce_df_memory:
                    df_all = reduce_mem_usage(df_all)
                if n_users:
                    train_users = set(
                        np.random.choice(df_all[df_all["is_test"] == 0].user_id.unique(), n_users, replace=False)
                    )
                    # select a frozen set of users' clickouts for validation
                    df_all = df_all[(df_all.user_id.isin(train_users)) | (df_all.is_val == 1)]
            print("Training on {} users".format(df_all["user_id"].nunique()))
            print("Training data shape", df_all.shape)
        with timer("splitting timebased"):
            df_train = df_all[df_all["is_val"] == 0]
            df_val = df_all[df_all["is_val"] == 1]
            print("df_train shape", df_train.shape)
            print("df_val shape", df_val.shape)
        return df_train, df_val

    def load_train_test(self, n_users):
        with timer("Reading training and testing data"):
            df_all = pd.read_csv(self.datapath)
            if self.reduce_df_memory:
                df_all = reduce_mem_usage(df_all)
            df_test = df_all[df_all["is_test"] == 1]
            if n_users:
                train_users = set(
                    np.random.choice(df_all[df_all["is_test"] == 0].user_id.unique(), n_users, replace=False)
                )
                # always include all the users from the test set
                train_users |= set(df_all[df_all["is_test"] == 1].user_id.unique())
                df_all = df_all[(df_all.user_id.isin(train_users)) & (df_all.is_test == 0)]
            print("Training on {} users".format(df_all["user_id"].nunique()))
            print("Training data shape", df_all.shape)
        return df_all, df_test
