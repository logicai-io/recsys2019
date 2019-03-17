import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

print("item_metadata to parquet...")
item_meta = pd.read_csv("./events_sorted_trans.csv")
meta_table = pa.Table.from_pandas(item_meta)
pq.write_table(meta_table, "./events_sorted_trans.parquet")
