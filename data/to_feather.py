import glob
import os
import matplotlib.pyplot as plt
import time

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.feather as pf

from pprint import pprint

files = sorted(glob.glob('./*.csv'))
print('files:')
pprint(files)

print('train to.feather...')
train_df = pd.read_csv('./train.csv')
pf.write_feather(train_df, './train.feather')

print('test to.feather...')
test_df = pd.read_csv('./test.csv')
pf.write_feather(test_df, './test.feather')

print('item_metadata to.feather...')
item_meta = pd.read_csv('./item_metadata.csv')
pf.write_feather(item_meta, './item_metadata.feather')
