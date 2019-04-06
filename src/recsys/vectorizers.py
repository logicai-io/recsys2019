from recsys.transformers import (
    FeatureEng,
    LagNumericalFeaturesWithinGroup,
    MinimizeNNZ,
    PandasToNpArray,
    PandasToRecords,
    RankFeatures,
)
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
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
    "user_rank_preference",
    "user_session_rank_preference",
    "user_impression_rank_preference",
    "avg_price_similarity",
    "avg_price_similarity_to_interacted_items",
    "avg_price_similarity_to_interacted_session_items",
    "clickout_item_item_last_timestamp",
    "interaction_item_image_item_last_timestamp",
    "filter_selection_count",
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
    "user_rank_preference",
    "user_session_rank_preference",
    "user_impression_rank_preference",
    "avg_price_similarity",
    "avg_price_similarity_to_interacted_items",
    "avg_price_similarity_to_interacted_session_items",
]
categorical_features_py = ["device", "platform", "last_sort_order", "last_filter_selection", "country", "hotel_cat"]


def make_vectorizer_1(
    categorical_features=categorical_features_py,
    numerical_features=numerical_features_py,
    numerical_features_for_ranking=numerical_features_for_ranking_py,
):
    return make_pipeline(
        FeatureEng(),
        ColumnTransformer(
            [
                (
                    "numerical",
                    make_pipeline(PandasToNpArray(), SimpleImputer(strategy="mean"), StandardScaler()),
                    numerical_features,
                ),
                (
                    "numerical_context",
                    make_pipeline(LagNumericalFeaturesWithinGroup(), MinimizeNNZ()),
                    numerical_features + ["clickout_id"],
                ),
                ("categorical", make_pipeline(PandasToRecords(), DictVectorizer()), categorical_features),
                (
                    "numerical_ranking",
                    make_pipeline(RankFeatures(), MinimizeNNZ()),
                    numerical_features_for_ranking + ["clickout_id"],
                ),
                ("properties", CountVectorizer(tokenizer=lambda x: x, lowercase=False, min_df=2), "properties"),
                (
                    "last_filter",
                    CountVectorizer(
                        preprocessor=lambda x: "UNK" if x != x else x, tokenizer=lambda x: x.split("|"), min_df=2
                    ),
                    "last_filter",
                ),
                ("last_10_actions", CountVectorizer(ngram_range=(3, 3), tokenizer=list, min_df=2), "last_10_actions"),
                ("last_event_ts_dict", DictVectorizer(), "last_event_ts_dict"),
                # ("user_rank_dict", DictVectorizer(), "user_rank_dict"),
                # ("user_session_rank_dict", DictVectorizer(), "user_session_rank_dict"),
            ]
        ),
    )


def make_vectorizer_2(
    numerical_features=numerical_features_py,
    numerical_features_for_ranking=numerical_features_for_ranking_py,
    categorical_features=categorical_features_py,
):
    return make_pipeline(
        FeatureEng(),
        ColumnTransformer(
            [
                (
                    "numerical",
                    make_pipeline(PandasToNpArray(), SimpleImputer(strategy="mean"), KBinsDiscretizer()),
                    numerical_features,
                ),
                (
                    "numerical_context",
                    make_pipeline(LagNumericalFeaturesWithinGroup(), MinimizeNNZ()),
                    numerical_features + ["clickout_id"],
                ),
                ("categorical", make_pipeline(PandasToRecords(), DictVectorizer()), categorical_features),
                (
                    "numerical_ranking",
                    make_pipeline(RankFeatures(), MinimizeNNZ()),
                    numerical_features_for_ranking + ["clickout_id"],
                ),
                ("properties", CountVectorizer(tokenizer=lambda x: x, lowercase=False, min_df=2), "properties"),
                (
                    "last_filter",
                    CountVectorizer(
                        preprocessor=lambda x: "UNK" if x != x else x, tokenizer=lambda x: x.split("|"), min_df=2
                    ),
                    "last_filter",
                ),
                ("last_10_actions", CountVectorizer(ngram_range=(3, 3), tokenizer=list, min_df=2), "last_10_actions"),
                ("last_event_ts_dict", DictVectorizer(), "last_event_ts_dict"),
            ]
        ),
    )
