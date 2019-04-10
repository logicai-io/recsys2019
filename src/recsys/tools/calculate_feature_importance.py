import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from recsys.metric import mrr_fast


@click.command()
@click.option("--src", type=str, required=True, help="File path with features")
@click.option("--limit", type=int, default=None, help="Limit number of rows")
@click.option("--dst", type=str, default=None, help="Save results")
def main(src, dst, limit):
    df = pd.read_csv(src, nrows=limit)
    # df = df[df["is_impression_the_same"] == True]
    results = []
    for col in tqdm(df.columns):
        if df[col].dtype in [np.int, np.float] and col != "was_clicked":
            results.append((col, mrr_fast(df, col)))
            df[col + "_rank"] = df.groupby("clickout_id")[col].rank("max", ascending=False)
            mrr_rank = mrr_fast(df, col + "_rank")
            df[col + "_rank_rev"] = df.groupby("clickout_id")[col].rank("max", ascending=True)
            mrr_rank_rev = mrr_fast(df, col + "_rank_rev")
            results.append((col + "_rank", max(mrr_rank, mrr_rank_rev)))
    results_df = pd.DataFrame.from_records(results, columns=["col", "mrr"])
    results_df.sort_values("mrr", ascending=False, inplace=True)
    print(results_df)
    if dst:
        results_df.to_csv(dst, index=False)


if __name__ == "__main__":
    main()
