import joblib
from recsys.lazy_count_vectorizer import LazyCountVectorizer
from recsys.transformers import (
    FeatureEng,
    LagNumericalFeaturesWithinGroup,
    PandasToNpArray,
    PandasToRecords,
    RankFeatures,
    ToCSR,
    PATH_TO_IMM,
)
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

numerical_features_py = [
    "rank",
    "item_id",
    "price",
    "price_vs_max_price",
    "price_vs_mean_price",
    "last_poi_item_clicks",
    "last_poi_item_impressions",
    "last_poi_item_ctr",
    "user_item_ctr",
    "last_item_index",
    "clickout_user_item_clicks",
    "clickout_item_clicks",
    "clickout_item_impressions",
    "was_interaction_img",
    "interaction_img_freq",
    "was_interaction_deal",
    "interaction_deal_freq",
    "was_interaction_info",
    "interaction_info_freq",
    "was_item_searched",
    "was_interaction_rating",
    "interaction_rating_freq",
    "interaction_img_diff_ts",
    "identical_impressions_item_clicks",
    "clickout_item_platform_clicks",
    "is_impression_the_same",
    "clicked_before",
    "country_eq_platform",
    "item_similarity_to_last_clicked_item",
    "avg_similarity_to_interacted_items",
    "avg_similarity_to_interacted_session_items",
    "hour",
    "n_properties",
    "rating",
    "stars",
    "last_index_1",
    "last_index_2",
    "last_index_3",
    "last_index_4",
    "last_index_5",
    "last_index_diff_1",
    "last_index_diff_2",
    "last_index_diff_3",
    "last_index_diff_4",
    "last_index_diff_5",
    "n_consecutive_clicks",
]
numerical_features_for_ranking_py = [
    "price",
    "last_poi_item_clicks",
    "last_poi_item_impressions",
    "last_poi_item_ctr",
    "user_item_ctr",
    "clickout_user_item_clicks",
    "clickout_item_clicks",
    "clickout_item_impressions",
    "interaction_img_freq",
    "interaction_deal_freq",
    "interaction_info_freq",
    "interaction_rating_freq",
    "identical_impressions_item_clicks",
    "item_id",
    "clickout_item_platform_clicks",
    "item_similarity_to_last_clicked_item",
    "avg_similarity_to_interacted_items",
    "avg_similarity_to_interacted_session_items",
    "n_properties",
    "rating",
    "stars",
    "n_consecutive_clicks",
]
categorical_features_py = ["device", "platform", "last_sort_order", "last_filter_selection", "country", "hotel_cat"]

feature_eng = FeatureEng()


def make_vectorizer_1(
    categorical_features=categorical_features_py,
    numerical_features=numerical_features_py,
    numerical_features_for_ranking=numerical_features_for_ranking_py,
):
    return make_pipeline(
        feature_eng,
        ColumnTransformer(
            [
                (
                    "numerical",
                    make_pipeline(PandasToNpArray(), SimpleImputer(strategy="mean"), StandardScaler()),
                    numerical_features,
                ),
                ("numerical_context", LagNumericalFeaturesWithinGroup(), numerical_features + ["clickout_id"]),
                ("categorical", make_pipeline(PandasToRecords(), FeatureHasher(n_features=2048)), categorical_features),
                ("numerical_ranking", RankFeatures(), numerical_features_for_ranking + ["clickout_id"]),
                (
                    "properties",
                    HashingVectorizer(tokenizer=lambda x: x, lowercase=False, n_features=2048),
                    "properties",
                ),
                (
                    "last_filter",
                    HashingVectorizer(
                        preprocessor=lambda x: "UNK" if x != x else x, tokenizer=lambda x: x.split("|"), n_features=2048
                    ),
                    "last_filter",
                ),
                (
                    "last_10_actions",
                    HashingVectorizer(ngram_range=(1, 5), tokenizer=list, n_features=2048),
                    "last_10_actions",
                ),
                ("last_event_ts_dict", FeatureHasher(n_features=2048), "last_event_ts_dict"),
            ]
        ),
    )


def make_vectorizer_2(
    numerical_features=numerical_features_py,
    numerical_features_for_ranking=numerical_features_for_ranking_py,
    categorical_features=categorical_features_py,
):
    properties_map = joblib.load(PATH_TO_IMM)
    return make_pipeline(
        feature_eng,
        ColumnTransformer(
            [
                (
                    "numerical",
                    make_pipeline(PandasToNpArray(), SimpleImputer(strategy="mean"), KBinsDiscretizer()),
                    numerical_features,
                ),
                ("numerical_context", LagNumericalFeaturesWithinGroup(), numerical_features + ["clickout_id"]),
                ("categorical", make_pipeline(PandasToRecords(), FeatureHasher(n_features=2048)), categorical_features),
                (
                    "numerical_ranking",
                    make_pipeline(RankFeatures(), StandardScaler()),
                    numerical_features_for_ranking + ["clickout_id"],
                ),
                (
                    "properties",
                    HashingVectorizer(tokenizer=lambda x: x, lowercase=False, n_features=2048),
                    "properties",
                ),
                (
                    "last_filter",
                    HashingVectorizer(
                        preprocessor=lambda x: "UNK" if x != x else x, tokenizer=lambda x: x.split("|"), n_features=2048
                    ),
                    "last_filter",
                ),
                (
                    "last_10_actions",
                    HashingVectorizer(ngram_range=(1, 5), tokenizer=list, n_features=128),
                    "last_10_actions",
                ),
                ("last_event_ts_dict", FeatureHasher(n_features=2048), "last_event_ts_dict"),
            ]
        ),
    )
