import pandas as pd
import joblib
import glob
import numpy as np

from recsys.metric import mrr_fast

click_preds = []
for fn in glob.glob("blend/*.joblib"):
    _, loss, preds = joblib.load(fn)
    if loss > 0.6372:
        print(loss)
        click_preds.append(preds["click_proba"].values)

preds_merged = np.vstack(click_preds).T
print(preds_merged.shape)
preds["merged"] = preds_merged.mean(axis=1)
print(mrr_fast(preds, "merged"))
