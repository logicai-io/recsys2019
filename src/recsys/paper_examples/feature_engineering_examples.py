import pandas as pd
from recsys.transformers import RankFeatures, LagNumericalFeaturesWithinGroup
from recsys.vectorizers import identity
from sklearn.pipeline import make_union
from sklearn.preprocessing import FunctionTransformer

df = pd.DataFrame.from_records(
    [
        {"clickout_id": 1, "price": 100},
        {"clickout_id": 1, "price": 80},
        {"clickout_id": 1, "price": 200},
        {"clickout_id": 1, "price": 150},
        {"clickout_id": 1, "price": 50},
    ]
)


for trans in [RankFeatures(ascending=True, drop_clickout_id=False)]:
    df = trans.fit_transform(df)
print(df)

df.to_csv("price_transformations.csv", index=False)
