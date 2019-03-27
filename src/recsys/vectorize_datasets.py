import gc
import glob
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from recsys.vectorizers import make_vectorizer_1, make_vectorizer_2
from scipy import sparse
from scipy.sparse import load_npz, save_npz


class VectorizeChunks:
    def __init__(self, vectorizer, input_files, output_folder, join_only=False, n_jobs=-2):
        self.vectorizer = vectorizer
        self.input_files = input_files
        self.output_folder = output_folder
        self.join_only = join_only
        self.n_jobs = n_jobs

    def vectorize_all(self):
        # fit vectorizers using the last chunk (I guess the test distribution is more important than training)
        if not self.join_only:
            df = pd.read_csv(sorted(glob.glob(self.input_files))[-1])
            self.vectorizer = self.vectorizer()
            self.vectorizer.fit(df)
        filenames = Parallel(n_jobs=self.n_jobs)(
            delayed(self.vectorize_one)(fn) for fn in sorted(glob.glob(self.input_files))
        )
        metadata_fns, csr_fns = list(zip(*filenames))
        self.save_to_one_file_metadata(metadata_fns)
        self.save_to_one_flie_csrs(csr_fns)

    def save_to_one_file_metadata(self, fns):
        dfs = [pd.read_hdf(os.path.join(self.output_folder, "chunks", fn), key="data") for fn in fns]
        df = pd.concat(dfs, axis=0)
        df.to_hdf(os.path.join(self.output_folder, "events_sorted_trans.h5"), key="data", mode="w")
        gc.collect()

    def save_to_one_flie_csrs(self, fns):
        matc = None
        for fn in fns:
            print(fn)
            mat = load_npz(os.path.join(self.output_folder, "chunks", fn)).astype(np.float16)
            if matc is not None:
                matc = sparse.vstack([matc, mat])
            else:
                matc = mat
            gc.collect()
        save_npz(os.path.join(self.output_folder, "events_sorted_trans_features.npz"), matc.astype(np.float32))
        gc.collect()

    def vectorize_one(self, fn):
        print(fn)
        fname_h5 = fn.split("/")[-1].replace(".csv", ".h5")
        fname_npz = fn.split("/")[-1].replace(".csv", ".npz")

        if self.join_only:
            return (fname_h5, fname_npz)

        df = pd.read_csv(fn)
        mat = self.vectorizer.transform(df)

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
        ].to_hdf(os.path.join(self.output_folder, "chunks", fname_h5), key="data", mode="w")

        save_npz(os.path.join(self.output_folder, "chunks", fname_npz), mat)

        gc.collect()

        return (fname_h5, fname_npz)


if __name__ == "__main__":
    vectorize_chunks = VectorizeChunks(
        vectorizer=lambda: make_vectorizer_1(),
        input_files="../../data/events_sorted_trans_chunks/raw_csv/events_sorted_trans_*.csv",
        output_folder="../../data/events_sorted_trans_chunks/vectorizer_1/",
        n_jobs=12,
    )
    vectorize_chunks.vectorize_all()

    vectorize_chunks = VectorizeChunks(
        vectorizer=lambda: make_vectorizer_2(),
        input_files="../../data/events_sorted_trans_chunks/raw_csv/events_sorted_trans_*.csv",
        output_folder="../../data/events_sorted_trans_chunks/vectorizer_2/",
        n_jobs=12,
    )
    vectorize_chunks.vectorize_all()
