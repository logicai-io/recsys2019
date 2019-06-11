import glob

import joblib
import pandas as pd
from recsys.vectorizers import make_vectorizer_1

if __name__ == "__main__":
    input_files = "../../data/proc/raw_csv/*.csv"
    vectorizer = make_vectorizer_1()
    df = pd.read_csv(sorted(glob.glob(input_files))[-1])
    vectorizer.fit(df)
    joblib.dump(vectorizer, "../../data/proc/vectorizer_1/vectorizer.joblib")
