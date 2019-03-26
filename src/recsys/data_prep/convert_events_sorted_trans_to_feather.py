import pandas as pd

import pyarrow as pa
import pyarrow.feather as pf

print("events_sorted_trans to feather...")
events = pd.read_csv("./events_sorted_trans.csv")
pf.write_feather(events, "./events_sorted_trans.feather")
