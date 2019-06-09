import click
import glob
import os

import joblib
import pandas as pd
from recsys.vectorizers import make_vectorizer_1
from scipy.sparse import save_npz


@click.command()
@click.option("--vectorizer_path", type=str, help="Path to vectorizer")
@click.option("--input", type=str, help="Path to input")
@click.option("--output_folder", type=str, help="Path to output")
def main(vectorizer_path, input, output_folder):
    fname_h5 = input.split("/")[-1].replace(".csv", ".h5")
    fname_npz = input.split("/")[-1].replace(".csv", ".npz")
    df = pd.read_csv(input)
    vectorizer = joblib.load(vectorizer_path)
    mat = vectorizer.transform(df)
    df[
        [
            "user_id",
            "session_id",
            "platform",
            "device",
            "city",
            "timestamp",
            "step",
            "clickout_id",
            "item_id",
            "src",
            "is_test",
            "is_val",
            "was_clicked",
        ]
    ].to_hdf(os.path.join(output_folder, "chunks", fname_h5), key="data", mode="w")
    save_npz(os.path.join(output_folder, "chunks", fname_npz), mat)

if __name__ == '__main__':
    main()
