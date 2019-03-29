import numpy as np
import pandas as pd

RATING_MAP = {"Satisfactory Rating": 1, "Good Rating": 2, "Very Good Rating": 3, "Excellent Rating": 4}
STAR_MAP = {"1 Star": 1, "2 Star": 2, "3 Star": 3, "4 Star": 4, "5 Star": 5}
HOTEL_CAT = {
    "Hotel": "hotel",
    "Resort": "resort",
    "Hostal (ES)": "hostal",
    "Motel": "motel",
    "House / Apartment": "house",
}
IMPORTANT_FEATURES = [
    "Free WiFi (Combined)",
    "Swimming Pool (Combined Filter)",
    "Car Park",
    "Serviced Apartment",
    "Air Conditioning",
    "Spa (Wellness Facility)",
    "Pet Friendly",
    "All Inclusive (Upon Inquiry)",
]


def densify(d, properties):
    values = [None] * properties.shape[0]
    for i, p in enumerate(properties):
        for k in d:
            if k in p:
                values[i] = d[k]
    return values


def normalize_feature_name(name):
    return name.replace(" ", "_").lower()


if __name__ == "__main__":
    df = pd.read_csv("../../../data/item_metadata.csv")
    df["properties"] = df["properties"].str.split("|").map(set)
    df["n_properties"] = df["properties"].map(len)
    df["rating"] = densify(RATING_MAP, df["properties"])
    df["stars"] = densify(STAR_MAP, df["properties"])
    df["hotel_cat"] = densify(HOTEL_CAT, df["properties"])
    for f in IMPORTANT_FEATURES:
        df[normalize_feature_name(f)] = df["properties"].map(lambda p: f in p).astype(np.int)
    df.drop("properties", axis=1).to_csv("../../../data/item_metadata_dense.csv", index=False)
