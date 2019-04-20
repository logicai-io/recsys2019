from recsys.transformers import (
    FeatureEng,
    LagNumericalFeaturesWithinGroup,
    MinimizeNNZ,
    NormalizeClickSequence,
    PandasToNpArray,
    PandasToRecords,
    RankFeatures,
    SparsityFilter,
)
from recsys.utils import str_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, FunctionTransformer

numerical_features_info = [
    ("avg_price_similarity", True),
    ("avg_price_similarity_to_interacted_items", True),
    ("avg_price_similarity_to_interacted_session_items", True),
    ("avg_similarity_to_interacted_items", True),
    ("avg_similarity_to_interacted_session_items", True),
    ("clicked_before", False),
    ("clickout_item_clicks", True),
    ("clickout_item_impressions", True),
    ("clickout_user_item_clicks", True),
    ("user_item_ctr", True),
    ("clickout_item_item_last_timestamp", False),
    ("clickout_item_platform_clicks", True),
    ("clickout_prob_time_position_offset", True),
    ("country_eq_platform", False),
    ("fake_last_index_1", False),
    ("fake_last_index_2", False),
    ("fake_last_index_3", False),
    ("fake_last_index_4", False),
    ("fake_last_index_5", False),
    ("fake_last_index_diff_1", False),
    ("fake_last_index_diff_2", False),
    ("fake_last_index_diff_3", False),
    ("fake_last_index_diff_4", False),
    ("fake_last_index_diff_5", False),
    ("fake_n_consecutive_clicks", True),
    ("filter_selection_count", False),
    ("hour", False),
    ("identical_impressions_item_clicks", True),
    ("identical_impressions_item_clicks2", False),
    ("interaction_deal_freq", True),
    ("interaction_img_diff_ts", False),
    ("interaction_img_freq", True),
    ("interaction_info_freq", True),
    ("interaction_item_image_item_last_timestamp", False),
    ("interaction_rating_freq", True),
    ("is_impression_the_same", False),
    ("item_id", True),
    ("item_similarity_to_last_clicked_item", True),
    ("last_index_1", False),
    ("last_index_2", False),
    ("last_index_3", False),
    ("last_index_4", False),
    ("last_index_5", False),
    ("last_index_diff_1", False),
    ("last_index_diff_2", False),
    ("last_index_diff_3", False),
    ("last_index_diff_4", False),
    ("last_index_diff_5", False),
    ("last_item_fake_index", False),
    ("last_item_index", False),
    ("last_poi_item_clicks", True),
    ("last_poi_item_ctr", True),
    ("last_poi_item_impressions", True),
    ("last_price_diff", True),
    ("n_consecutive_clicks", True),
    ("n_properties", True),
    ("num_pois", True),
    ("poi_avg_similarity_to_interacted_items", True),
    ("poi_item_similarity_to_last_clicked_item", True),
    ("price", True),
    ("price_vs_max_price", False),
    ("price_vs_mean_price", False),
    ("rank", False),
    ("rating", True),
    ("stars", True),
    ("user_fake_rank_preference", True),
    ("user_impression_rank_preference", True),
    ("user_rank_preference", True),
    ("user_session_rank_preference", True),
    ("was_interaction_deal", False),
    ("was_interaction_img", False),
    ("was_interaction_info", False),
    ("was_interaction_rating", False),
    ("was_item_searched", False),
]

numerical_features_for_ranking_py = [f for f, rank in numerical_features_info if rank]
numerical_features_py = [f for f, rank in numerical_features_info]
categorical_features_py = [
    "device",
    "platform",
    "last_sort_order",
    "last_filter_selection",
    "country",
    "hotel_cat",
    "city",
    "last_poi",
]
numerical_features_offset_2 = ["was_interaction_info", "was_interaction_img", "last_index_diff_5"]


def make_vectorizer_1(
    categorical_features=categorical_features_py,
    numerical_features=numerical_features_py,
    numerical_features_offset_2=numerical_features_offset_2,
    numerical_features_for_ranking=numerical_features_for_ranking_py,
):
    return make_pipeline(
        FeatureEng(),
        # NormalizeClickSequence(),
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
                (
                    "numerical_context_offset_2",
                    make_pipeline(LagNumericalFeaturesWithinGroup(offset=2), MinimizeNNZ()),
                    numerical_features_offset_2 + ["clickout_id"],
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
                ("last_poi_bow", CountVectorizer(min_df=5), "last_poi"),
                ("last_event_ts_dict", DictVectorizer(), "last_event_ts_dict"),
            ]
        ),
    )
