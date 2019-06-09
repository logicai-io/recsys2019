import glob

import subprocess
import os
from functools import partial
from random import random
from time import sleep

from multiprocessing.pool import Pool
from recsys.vectorizers import VectorizeChunks
import time

def run_one(vectorizer_path, output_folder, fn):
    # sleep for some time
    sleep(int(random()*10*60))
    print(fn, "started")
    args = ["python", "vectorizer_transform.py",
            "--vectorizer_path", vectorizer_path,
            "--input", fn,
            "--output_folder", output_folder]
    p = subprocess.Popen(args)
    p.wait()
    print(fn, "finished")
    return 1

if __name__ == '__main__':
    vectorizer_path = "../../data/proc/vectorizer_1/vectorizer.joblib"
    input_files = "../../data/proc/raw_csv/*.csv"
    output_folder = "../../data/proc/vectorizer_1_v2/"
    ps = []

    with Pool(10) as pool:
        m = pool.imap_unordered(partial(run_one, vectorizer_path, output_folder), sorted(glob.glob(input_files)))
        results = list(m)

    # for fn in sorted(glob.glob(input_files)):
    #     print(fn)
    #     args = ["python", "vectorizer_transform.py",
    #             "--vectorizer_path", vectorizer_path,
    #             "--input", fn,
    #             "--output_folder", output_folder]
    #     p = subprocess.Popen(args)
    #     ps.append(p)
    #     sleep(60)
    #
    # for p in ps:
    #     p.wait()

    # vectorize_chunks = VectorizeChunks(
    #     vectorizer=None,
    #     input_files=input_files,
    #     output_folder=output_folder,
    #     n_jobs=6,
    #     join_only=True
    # )
    # vectorize_chunks.vectorize_all()
