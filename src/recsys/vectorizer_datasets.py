import glob
import os

import pandas as pd
from joblib import Parallel, delayed
from recsys.vectorizers_par import make_vectorizer_1, make_vectorizer_2
from scipy.sparse import save_npz


class VectorizeChunks:
    def __init__(self, vectorizer, input_files, output_folder, n_jobs=-2):
        self.vectorizer = vectorizer
        self.input_files = input_files
        self.output_folder = output_folder
        self.n_jobs = n_jobs

    def vectorize_all(self):
        _ = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self.vectorize_one)(fn) for fn in glob.glob(self.input_files)
        )

    def vectorize_one(self, fn):
        print(fn)
        df = pd.read_csv(fn)
        mat = self.vectorizer.fit_transform(df)

        fname_csv = fn.split("/")[-1]
        fname_npz = fn.split("/")[-1].replace(".csv", ".npz")

        df[["user_id", "session_id", "timestamp", "step", "clickout_id", "src", "is_test", "is_val"]].to_csv(
            os.path.join(self.output_folder, fname_csv), index=False
        )

        save_npz(os.path.join(self.output_folder, fname_npz), mat)

        return fn


if __name__ == "__main__":
    vectorize_chunks = VectorizeChunks(
        vectorizer=make_vectorizer_1(),
        input_files="../../data/events_sorted_trans_chunks/raw_csv/events_sorted_trans_*.csv",
        output_folder="../../data/events_sorted_trans_chunks/vectorizer_1/chunks/",
        n_jobs=8,
    )
    vectorize_chunks.vectorize_all()

    vectorize_chunks = VectorizeChunks(
        vectorizer=make_vectorizer_2(),
        input_files="../../data/events_sorted_trans_chunks/raw_csv/events_sorted_trans_*.csv",
        output_folder="../../data/events_sorted_trans_chunks/vectorizer_2/chunks/",
        n_jobs=8,
    )
    vectorize_chunks.vectorize_all()
