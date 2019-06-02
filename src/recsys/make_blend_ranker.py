import pandas as pd
from recsys.log_utils import get_logger
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts
from scipy.optimize import fmin
from lightgbm import LGBMRanker
from recsys.utils import group_lengths, timer, get_git_hash

logger = get_logger()

logger.info("Staring blending")

p_lgbr = pd.read_csv("predictions/model_val_92dc89e0765dbc875a8052ae0dbdd7e5ee4271bc.csv")
p_lgbr2 = pd.read_csv("predictions/model_val_365e3484ea47c9c0e5dab466da1a441656cf13d9.csv")
p_lgbr3 = pd.read_csv("predictions/model_val_d2c1eff343a5048da32abf64f3a5daf0f7bc0025.csv")
X = pd.DataFrame({"a": p_lgbr["click_proba"], "b": p_lgbr2["click_proba"], "c": p_lgbr3["click_proba"]})
y = p_lgbr["was_clicked"]
g = p_lgbr["clickout_id"]

cutoff = X.shape[0] // 2
X_tr, X_te, y_tr, y_te, g_tr, g_te = X[:cutoff], X[cutoff:], y[:cutoff], y[cutoff:], g[:cutoff], g[cutoff:]

model = LGBMRanker(n_estimators=3)
model.fit(X_tr, y_tr, group=group_lengths(g_tr))

pred = model.predict(X)
p_lgbr["pred"] = pred
print(mrr_fast(p_lgbr.iloc[cutoff:], "click_proba"))
print(mrr_fast(p_lgbr.iloc[cutoff:], "pred"))

assert False
p_lgbr = pd.read_csv("predictions/model_submit_92dc89e0765dbc875a8052ae0dbdd7e5ee4271bc.csv")
p_lgbr2 = pd.read_csv("predictions/model_submit_365e3484ea47c9c0e5dab466da1a441656cf13d9.csv")
p_lgbr3 = pd.read_csv("predictions/model_submit_d2c1eff343a5048da32abf64f3a5daf0f7bc0025.csv")
final = p_lgbr.copy()
final["click_proba"] = a * p_lgbr["click_proba"] + b * p_lgbr2["click_proba"] + c * p_lgbr3["click_proba"]
_, submission_df = group_clickouts(final)
mrr_str = str(mrr).split(".")[1][:4]
save_as = f"submissions/submission_{mrr_str}.csv"
logger.info(f"Saving submission file {save_as}")
submission_df.to_csv(save_as, index=False)
