import glob
import os
import matplotlib.pyplot as plt
import time

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.feather as pf

from pprint import pprint

print("train to.feather...")
train_df = pd.read_csv("../../../data/train.csv")
pf.write_feather(train_df, "../../../data/train.feather")

print("test to.feather...")
test_df = pd.read_csv("../../../data/test.csv")
pf.write_feather(test_df, "../../../data/test.feather")

print("item_metadata to.feather...")
item_meta = pd.read_csv("../../../data/item_metadata.csv")
pf.write_feather(item_meta, "../../../data/item_metadata.feather")
