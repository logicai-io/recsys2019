import pathlib
import warnings

import click
import numpy as np
from lightgbm import LGBMClassifier, LGBMRanker
from recsys.model_utils import Model, ModelTrain
from recsys.utils import timer
from recsys.vectorizers import make_vectorizer_1, make_vectorizer_2
from recsys.vectorizers_scala import make_vectorizer_1_scala

warnings.filterwarnings("ignore")

np.random.seed(0)


def info(action, load_feather, n_jobs, n_users, reduce_df_memory):
    print(f"n_users={n_users}")
    print(f"action={action}")
    print(f"n_jobs={n_jobs}")
    print(f"reduce_df_memory={reduce_df_memory}")
    print(f"load_feather={load_feather}")


@click.command()
@click.option("--data", type=str, default="python", help="Generated data")
@click.option("--n_users", type=int, default=None, help="Number of users to user for training")
@click.option("--n_trees", type=int, default=100, help="Number of trees for lightgbm models")
@click.option("--n_jobs", type=int, default=-2, help="Number of cores to run models on")
@click.option("--n_debug", type=int, default=None, help="Number of rows to use for debuging")
@click.option("--action", type=str, default="validate", help="What to do: validate/submit")
@click.option("--reduce_df_memory", type=bool, default=True, help="Aggresively reduce DataFrame memory")
@click.option("--load_feather", type=bool, default=False, help="Use .feather or .csv DataFrame")
def main(data, n_users, n_trees, n_jobs, n_debug, action, reduce_df_memory, load_feather):
    info(action, load_feather, n_jobs, n_users, reduce_df_memory)

    if data == "python":
        models = [
            Model(
                "data_py_lgbclas",
                make_vectorizer_1(),
                LGBMClassifier(n_estimators=n_trees, n_jobs=n_jobs),
                weight=1.0,
                is_prob=True,
            ),
            Model(
                "data_py_lgbrank",
                make_vectorizer_2(),
                LGBMRanker(n_estimators=n_trees, n_jobs=n_jobs),
                weight=0.2,
                is_prob=False,
            ),
        ]
        datapath = pathlib.Path().absolute().parents[1] / "data" / "events_sorted_trans.csv"
    elif data == "scala":
        models = [
            Model(
                "data_sc_lgbclas",
                make_vectorizer_1_scala(),
                LGBMClassifier(n_estimators=n_trees, n_jobs=n_jobs),
                weight=1.0,
                is_prob=True,
            ),
            Model(
                "data_sc_lgbrank",
                make_vectorizer_1_scala(),
                LGBMRanker(n_estimators=n_trees, n_jobs=n_jobs),
                weight=0.2,
                is_prob=False,
            ),
        ]
        datapath = pathlib.Path().absolute().parents[1] / "data" / "events_sorted_trans_scala.csv"

    trainer = ModelTrain(
        models=models, datapath=datapath, n_jobs=n_jobs, reduce_df_memory=reduce_df_memory, load_feather=load_feather
    )

    if action == "validate":
        with timer("validating models"):
            trainer.validate_models(n_users, n_debug)
    elif action == "submit":
        with timer("training full data models"):
            trainer.submit_models(n_users)


if __name__ == "__main__":
    main()
