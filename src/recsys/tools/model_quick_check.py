import click
import numpy as np
import pandas as pd
from lightgbm import LGBMRankerMRR, LGBMRanker
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast
from recsys.utils import group_lengths
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def augment_df(df, cols):
    for col in tqdm(cols):
        df[col + "_rank"] = df.groupby("clickout_id")[col].rank("max", ascending=False)
    return df


@click.command()
@click.option("--src", type=str, required=True, help="File path with features")
@click.option("--limit", type=int, default=None, help="Number of rows")
def main(src, limit):
    df = pd.read_csv(src, nrows=limit)
    print("Shape", df.shape)
    train, test = split_by_timestamp(df)
    train.fillna(-1000, inplace=True)
    test.fillna(-1000, inplace=True)

    tech_cols = [
        "user_id",
        "session_id",
        "timestamp",
        "step",
        "reference",
        "platform",
        "city",
        "device",
        "current_filters",
        "src",
        "is_test",
        "index_clicked",
        "item_id",
        "item_id_clicked",
        "was_clicked",
        "clickout_id",
        "rank",
    ]

    numerical_cols = [col for col in train.columns if df[col].dtype in [np.int, np.float] and col not in tech_cols]

    train = augment_df(train, numerical_cols)
    test = augment_df(test, numerical_cols)

    numerical_cols_rank = [c + "_rank" for c in numerical_cols]

    model = LGBMRanker()
    model.fit(train[numerical_cols + numerical_cols_rank], train["was_clicked"],
              group=group_lengths(train["clickout_id"].values))

    # fimp = pd.DataFrame(
    #     {"feature": numerical_cols + numerical_cols_rank, "importance": model.booster_.feature_importance()}
    # )
    # fimp.sort_values("importance", ascending=False, inplace=True)
    # print(fimp)
    #
    # fimp = pd.DataFrame(
    #     {"feature": numerical_cols + numerical_cols_rank, "importance": model.booster_.feature_importance("gain")}
    # )
    # fimp.sort_values("importance", ascending=False, inplace=True)
    # print(fimp)

    train["pred"] = model.predict(train[numerical_cols + numerical_cols_rank])
    print("Train:", mrr_fast(train, "pred"))
    test["pred"] = model.predict(test[numerical_cols + numerical_cols_rank])
    print("Test:", mrr_fast(test, "pred"))

    # model = LGBMRankerMRR()
    # model.fit(train[numerical_cols + numerical_cols_rank], train["was_clicked"],
    #           group=group_lengths(train["clickout_id"].values))
    #
    # fimp = pd.DataFrame(
    #     {"feature": numerical_cols + numerical_cols_rank, "importance": model.booster_.feature_importance()}
    # )
    # fimp.sort_values("importance", ascending=False, inplace=True)
    # print(fimp)
    #
    # fimp = pd.DataFrame(
    #     {"feature": numerical_cols + numerical_cols_rank, "importance": model.booster_.feature_importance("gain")}
    # )
    # fimp.sort_values("importance", ascending=False, inplace=True)
    # print(fimp)

    # train["pred"] = model.predict(train[numerical_cols + numerical_cols_rank])
    # print("Train:", mrr_fast(train, "pred"))
    # test["pred"] = model.predict(test[numerical_cols + numerical_cols_rank])
    # print("Test:", mrr_fast(test, "pred"))


if __name__ == "__main__":
    main()
