# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import joblib
from sklearn.feature_extraction import DictVectorizer
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

df = pd.read_csv("../data/events_sorted.csv", nrows=1000000)
imm = joblib.load("../data/item_metadata_map.joblib")

all_items = list(imm.keys())

df = df[df["action_type"] == "clickout item"]
df["reference"] = df["reference"].map(int)

df.head()

users_clicks = defaultdict(set)
users_impressions = defaultdict(set)
for i,row in tqdm(df.iterrows()):
    users_clicks[row["user_id"]].add(row["reference"])
    impressions = [int(item) for item in row["impressions"].split("|") if item != row["reference"]]
    for item in impressions:
        users_impressions[row["user_id"]].add(item)
    try:
        users_impressions[row["user_id"]].remove(row["reference"])
    except:
        pass

# +
pairs = []
for user, items in tqdm(users_clicks.items()):
    if len(items) >= 2:
        for a in items:
            for b in items:
                if a != b:
                    pairs.append((1, (a,b)))
                    
        for a in items:
            for b in users_impressions[user]:
                if a != b:
                    pairs.append((0, (a,b)))
                    
#                     random_a = random.choice(all_items)
#                     random_b = random.choice(all_items)   
#                     pairs.append((0, (random_a,random_b)))
                    
# -

allobs = []
ys = []
for y, (item_a, item_b) in tqdm(pairs):
    obs = {}
    for attr in imm[item_a] | imm[item_b]:
        obs["attr_{:04d}".format(attr)] = 1
#     for attr in imm[item_a] ^ imm[item_b]:
#         obs["xor_attr_{:04d}".format(attr)] = 1
    obs["jaccard"] = len(imm[item_a] & imm[item_b]) / (len(imm[item_a] | imm[item_b]) + 1)
    allobs.append(obs)
    ys.append(y)

vect = DictVectorizer()
mat = vect.fit_transform(allobs)
n_split = mat.shape[0] // 2
X_tr, X_te, y_tr, y_te = mat[:n_split,:], mat[n_split:,:], ys[:n_split], ys[n_split:] 

# +
logreg = LogisticRegression(penalty='l2', C=5)
# logreg = LGBMClassifier(n_estimators=100, n_jobs=-2)
logreg.fit(X_tr, y_tr)
pred = logreg.predict_proba(X_te)[:,1]
print(roc_auc_score(y_te, pred))

pred = logreg.predict_proba(X_tr)[:,1]
print(roc_auc_score(y_tr, pred))
# -

for attr_name, coef in zip(vect.get_feature_names(), list(logreg.coef_.reshape(-1))):
    print(attr_name, coef)

sum(ys), len(ys)


