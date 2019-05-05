from collections import defaultdict
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

df = pd.read_csv("../../../data/item_metadata.csv")

def split_by_pipe(text):
    return text.split("|")

tf = TfidfVectorizer(tokenizer=split_by_pipe)

properties_mat = tf.fit_transform(df["properties"])
item_id_ind = dict(zip(df["item_id"], range(df.shape[0])))
joblib.dump((item_id_ind, properties_mat), "../../../data/item_properties_tfidf.joblib", compress=3)