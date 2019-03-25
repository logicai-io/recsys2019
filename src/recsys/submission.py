import pandas as pd
from tqdm import tqdm


def group_clickouts(df):
    sessions_items = {}
    submission = []
    df = df.sort_values("click_proba", ascending=False)
    for session_id, session_df in df.groupby("clickout_id"):
        record = [session_df[col].to_list()[0] for col in ["user_id", "session_id", "timestamp", "step"]]
        submission.append(record + [" ".join(session_df.item_id.map(str).to_list())])
        sessions_items[session_id] = session_df.item_id.to_list()
    submission_df = pd.DataFrame(
        submission, columns="user_id,session_id,timestamp,step,item_recommendations".split(",")
    )
    return sessions_items, submission_df
