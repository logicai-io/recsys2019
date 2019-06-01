import pandas as pd
import numpy as np
from recsys.log_utils import get_logger
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts
from sklearn.linear_model import LogisticRegression
import glob

logger = get_logger()

logger.info("Staring blending")

X = []
for fn in glob.glob("predictions/model_2_val*.csv"):
    p_lgbr = pd.read_csv(fn)
    final = p_lgbr.copy()
    mrr = mrr_fast(final, "click_proba")
    if mrr > 0.6260:
        X.append(p_lgbr["click_proba"])
        logger.info(f"P {fn} MRR {mrr}")

X = np.vstack(X).T
y = p_lgbr["was_clicked"]
logreg = LogisticRegression()
logreg.fit(X, y)
pred = logreg.predict(X)

p_lgbr["final"] = pred
mrr_final = mrr_fast(p_lgbr, "final")
logger.info(f"Final blend {mrr}")

assert False

p_lgbr = pd.read_csv("predictions/model_2_submit.csv")
final = p_lgbr.copy()
final["click_proba"] = 1.0 * p_lgbr["click_proba"]
_, submission_df = group_clickouts(final)
mrr_str = str(mrr).split(".")[1][:4]
save_as = f"submissions/submission_{mrr_str}.csv"
logger.info(f"Saving submission file {save_as}")
submission_df.to_csv(save_as, index=False)
