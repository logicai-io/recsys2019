import gc
import glob
import os

import h5sparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from recsys.vectorizers import make_vectorizer_1, make_vectorizer_2
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
        df.to_hdf(os.path.join(self.output_folder, "meta.h5"), key="data", mode="w")
        gc.collect()

    def save_to_one_flie_csrs(self, fns):
        save_as = os.path.join(self.output_folder, "Xcsr.h5")
        os.unlink(save_as)
        h5f = h5sparse.File(save_as)
        first = True
        for fn in fns:
            print(fn)
            mat = load_npz(os.path.join(self.output_folder, "chunks", fn)).astype(np.float32)
            if first:
                h5f.create_dataset("matrix", data=mat, chunks=(10000000,), maxshape=(None,))
                first = False
            else:
                h5f["matrix"].append(mat)
            gc.collect()
        h5f.close()

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
        ].to_hdf(os.path.join(self.output_folder, "chunks", fname_h5), key="data", mode="w")

        save_npz(os.path.join(self.output_folder, "chunks", fname_npz), mat)

        gc.collect()

        return (fname_h5, fname_npz)


if __name__ == "__main__":
    vectorize_chunks = VectorizeChunks(
        vectorizer=lambda: make_vectorizer_1(),
        input_files="../../data/proc/raw_csv/*.csv",
        output_folder="../../data/proc/vectorizer_1/",
        n_jobs=10,
    )
    vectorize_chunks.vectorize_all()

    # vectorize_chunks = VectorizeChunks(
    #     vectorizer=lambda: make_vectorizer_2(),
    #     input_files="../../data/proc/raw_csv/*.csv",
    #     output_folder="../../data/proc/vectorizer_2/",
    #     n_jobs=10,
    # )
    # vectorize_chunks.vectorize_all()
