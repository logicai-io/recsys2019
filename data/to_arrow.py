import glob
import os
import matplotlib.pyplot as plt
import time

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from pprint import pprint

files = sorted(glob.glob('./*.csv'))
print('files:')
pprint(files)

print('train to parquet...')
train_df = pd.read_csv('./train.csv')
table = pa.Table.from_pandas(train_df)
pq.write_table(table, './train.parquet')

print('test to parquet...')
test_df = pd.read_csv('./test.csv')
test_table = pa.Table.from_pandas(test_df)
pq.write_table(test_table, './test.parquet')

print('item_metadata to parquet...')
item_meta = pd.read_csv('./item_metadata.csv')
meta_table = pa.Table.from_pandas(item_meta)
pq.write_table(meta_table, './item_metadata.parquet')
