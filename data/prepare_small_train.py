import pandas as pd

train = pd.read_csv("train.csv")

sample = train[train.timestamp < 1541068577]
sample.to_csv("train_sample_5.csv", index=False)

sample = train[train.timestamp < 1541037410]
sample.to_csv("train_sample_1.csv", index=False)
