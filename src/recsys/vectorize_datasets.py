import glob
import os
import gc

import pandas as pd
from joblib import Parallel, delayed
from recsys.vectorizers import make_vectorizer_1, make_vectorizer_2
from scipy import sparse
from scipy.sparse import load_npz, save_npz


class VectorizeChunks:
    def __init__(self, vectorizer, input_files, output_folder, n_jobs=-2):
        self.vectorizer = vectorizer
        self.input_files = input_files
        self.output_folder = output_folder
        self.n_jobs = n_jobs

    def vectorize_all(self):
        # fit vectorizers using the first chunk
        df = pd.read_csv(sorted(glob.glob(self.input_files))[-1])
        self.vectorizer.fit(df)

        filenames = Parallel(n_jobs=self.n_jobs)(                                         
            delayed(self.vectorize_one)(fn) for fn in sorted(glob.glob(self.input_files))
        )
        #filenames = [self.vectorize_one(fn) for fn in sorted(glob.glob(self.input_files))]
        dfs, mats = self.load_chunks(filenames)
        self.save_to_one_file(dfs, mats)

    def save_to_one_file(self, dfs, mats):
        df = pd.concat(dfs, axis=0)
        df.to_csv(os.path.join(self.output_folder, "events_sorted_trans.csv"), index=False)
        mat = sparse.vstack(mats)
        save_npz(os.path.join(self.output_folder, "events_sorted_trans_features.npz"), mat)

    def load_chunks(self, filenames):
        dfs = []
        mats = []
        for fn_csv, fn_npz in filenames:
            dfs.append(pd.read_csv(os.path.join(self.output_folder, "chunks", fn_csv)))
            mats.append(load_npz(os.path.join(self.output_folder, "chunks", fn_npz)))
        return dfs, mats

    def vectorize_one(self, fn):
        print(fn)
        df = pd.read_csv(fn)
        mat = self.vectorizer.transform(df)

        fname_csv = fn.split("/")[-1]
        fname_npz = fn.split("/")[-1].replace(".csv", ".npz")

        df[
            [
                "user_id",
                "session_id",
                "timestamp",
                "step",
                "clickout_id",
                "item_id",
                "src",
                "is_test",
                "is_val",
                "was_clicked",
            ]
        ].to_csv(os.path.join(self.output_folder, "chunks", fname_csv), index=False)

        save_npz(os.path.join(self.output_folder, "chunks", fname_npz), mat)

        gc.collect()

        return (fname_csv, fname_npz)


if __name__ == "__main__":
    vectorize_chunks = VectorizeChunks(
        vectorizer=make_vectorizer_1(),
        input_files="../../data/events_sorted_trans_chunks/raw_csv/events_sorted_trans_*.csv",
        output_folder="../../data/events_sorted_trans_chunks/vectorizer_1_parallel/",
        n_jobs=12,
    )
    vectorize_chunks.vectorize_all()

    #vectorize_chunks = VectorizeChunks(
    #    vectorizer=make_vectorizer_2(),
    #    input_files="../../data/events_sorted_trans_chunks/raw_csv/events_sorted_trans_*.csv",
    #    output_folder="../../data/events_sorted_trans_chunks/vectorizer_2/",
    #    n_jobs=16,
    #)
    #vectorize_chunks.vectorize_all()
