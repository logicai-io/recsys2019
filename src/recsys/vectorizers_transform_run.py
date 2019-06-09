import glob

import subprocess
import os
from time import sleep

from recsys.vectorizers import VectorizeChunks


if __name__ == '__main__':
    vectorizer_path = "../../data/proc/vectorizer_1/vectorizer.joblib"
    input_files = "../../data/proc/raw_csv/*.csv"
    output_folder = "../../data/proc/vectorizer_1/"
    ps = []
    for fn in sorted(glob.glob(input_files)):
        print(fn)
        args = ["python", "vectorizer_transform.py",
                "--vectorizer_path", vectorizer_path,
                "--input", fn,
                "--output_folder", output_folder]
        p = subprocess.Popen(args)
        ps.append(p)
        sleep(60)

    for p in ps:
        p.wait()

    vectorize_chunks = VectorizeChunks(
        vectorizer=None,
        input_files=input_files,
        output_folder=output_folder,
        n_jobs=6,
        join_only=True
    )
    vectorize_chunks.vectorize_all()
