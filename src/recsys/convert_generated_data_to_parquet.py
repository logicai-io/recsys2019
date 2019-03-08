import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

print('events to parquet...')
events = pd.read_csv('../../data/events_sorted_trans.csv')
table = pa.Table.from_pandas(events)
pq.write_table(table, '../../data/events_sorted_trans.parquet')
