# import joblib
import numpy as np
import pandas as pd

train = pq.read_table('./train.parquet').to_pandas()
train["src"] = "train"
train["is_test"] = 0
test = pq.read_table('./test.parquet').to_pandas()
test["src"] = "test"
test["is_test"] = (test["reference"].isnull() & (test["action_type"] == "clickout item")).astype(np.int)
events = pd.concat([train, test], axis=0)
events.sort_values(["timestamp", "user_id", "step"], inplace=True)

table = pa.Table.from_pandas(events)
pq.write_table(table, './events_sorted.parquet')
print('events saved.')
