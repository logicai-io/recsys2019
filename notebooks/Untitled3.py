# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd

df = pd.read_csv("../data/events_sorted.csv", nrows=1000000)
df = df[df["action_type"]=="clickout item"]
print(df.shape)


# +
# df = df[df["current_filters"].fillna("").str.find("Sort by Price")>0]
# -

def find_index(row):
    try:
        return row["impressions"].split("|").index(str(row["reference"]))
    except:
        return 0


df["index_clicked"] = df.apply(find_index, axis=1)
df["n_items"] = df["impressions"].map(lambda x: len(x.split("|")))

df["price_sorted"] = df["prices"].map(lambda x: sorted(map(int, x.split("|"))) == list(map(int, x.split("|"))))

df[df["n_items"]==25].groupby("price_sorted")["index_clicked"].mean()

df.groupby("price_sorted")["n_items"].count()


