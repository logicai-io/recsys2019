import gc
import glob
import os

import h5sparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from recsys.transformers import (
    FeatureEng,
    FeaturesAtAbsoluteRank,
    LagNumericalFeaturesWithinGroup,
    MinimizeNNZ,
    PandasToNpArray,
    PandasToRecords,
    RankFeatures,
    SanitizeSparseMatrix,
    SparsityFilter,
)
from recsys.utils import logger
from scipy.sparse import load_npz, save_npz
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    ("last_ts_diff_1", False),
    ("last_ts_diff_2", False),
    ("last_ts_diff_3", False),
    ("last_ts_diff_4", False),
    ("last_ts_diff_5", False),
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
    ("price_pct_by_city", True),
    ("price_pct_by_platform", True),
    ("session_start_ts", False),
    ("user_start_ts", False),
    ("session_count", False),
    ("step", False),
    ("clickout_step_rev", False),
    ("clickout_step", False),
    ("clickout_max_step", False),
    ("clickout_item_clicks_rank_weighted", False),
    ("clickout_item_impressions_rank_weighted", False),
    ("clickout_item_ctr_rank_weighted", False),
    ("item_clicks_when_last", False),
    ("item_impressions_when_last", False),
    ("item_ctr_when_last", False),
    ("item_average_seq_pos", False),
    ("similar_users_item_interaction", True),
    ("most_similar_item_interaction", True),
    ("datetime_hour", False),
    ("datetime_minute", False),
    ("datetime_local_hour", False),
    ("datetime_local_minute", False),
    ("price_rank_asc", True),
    ("price_rank_desc", True),
    ("price_rank_asc_pct", True),
    ("min_price", True),
    ("max_price", True),
    ("count_price", True),
    ("price_range", True),
    ("price_range_div", True),
    ("price_relative_to_min", True),
    ("item_stats_distance", True),
    ("item_stats_rating", True),
    ("item_stats_popularity", True),
    # ("graph_similarity_user_item_random_walk", True),
    # ("graph_similarity_user_item_clickout", True),
    # ("graph_similarity_user_item_search", True),
    # ("graph_similarity_user_item_interaction_info", True),
    # ("graph_similarity_user_item_interaction_img", True),
    # ("graph_similarity_user_item_intearction_deal", True),
    # ("graph_similarity_user_item_all_interactions", True),
    # ("graph_similarity_user_item_random_walk_resets", True),
    # ("avg_properties_similarity", True),
    # ("avg_properties_similarity_norm", True)
    # ("timestamp", False),
    # ("last_clickout_item_stats", True),
    # ("interaction_item_image_unique_num_by_session_id",True),
    # ("interaction_item_image_unique_num_by_timestamp",True),
    # ("clickout_item_unique_num_by_session_id",True),
    # ("clickout_item_unique_num_by_timestamp",True),
    # ("interaction_item_info_unique_num_by_timestamp",True),
    # ("interaction_item_info_unique_num_by_session_id",True),
    # ("search_for_item_unique_num_by_session_id",True),
    # ("search_for_item_unique_num_by_timestamp",True),
    # ("interaction_item_rating_unique_num_by_timestamp",True),
    # ("interaction_item_rating_unique_num_by_session_id",True),
    # ("interaction_item_deals_unique_num_by_timestamp",True),
    # ("interaction_item_deals_unique_num_by_session_id",True),
    # ("average_item_attention", True)
    # ("last_item_time_diff_same_user", False),
    # ("last_item_time_diff", False),
    ("click_sequence_min", False),
    ("click_sequence_max", False),
    ("click_sequence_min_norm", False),
    ("click_sequence_max_norm", False),
    ("click_sequence_len", False),
    ("click_sequence_sd", False),
    ("click_sequence_mean", False),
    ("click_sequence_mean_norm", False),
    ("click_sequence_gzip_len", False),
    ("click_sequence_entropy", False),
    ("fake_click_sequence_min", False),
    ("fake_click_sequence_max", False),
    ("fake_click_sequence_min_norm", False),
    ("fake_click_sequence_max_norm", False),
    ("fake_click_sequence_len", False),
    ("fake_click_sequence_sd", False),
    ("fake_click_sequence_mean", False),
    ("fake_click_sequence_mean_norm", False),
    ("fake_click_sequence_gzip_len", False),
    ("fake_click_sequence_entropy", False),
    ("clickout_counter_vs_interaction_counter_mean", False),
    ("mean_rank_counter_mean", False),
    ("identifier_counter_min_after", False),
    ("interaction_counter_pure", False),
    ("identifier_counter_max_after", False),
    ("identifier_counter_mean_before_vs_item", False),
    ("identifier_counter_prev_2_vs_item", False),
    ("interaction_counter_max_vs_item", False),
    ("interaction_counter_mean", False),
    ("mean_rank_counter_mean_after_vs_item", False),
    ("mean_rank_counter_rank_norm_after", False),
    ("mean_rank_counter_max_vs_item", False),
    ("mean_rank_counter_min", False),
    ("impression_counter_prev_1_vs_item", False),
    ("impression_counter_mean_before_vs_item", False),
    ("clickout_counter_vs_impression_counter_max_after", False),
    ("clickout_counter_vs_impression_counter_max_before", False),
    ("identifier_counter_rank_norm_after", False),
    ("impression_counter_rank_norm", False),
    ("impression_counter_mean_prev_3_vs_item", False),
    ("clickout_counter_vs_interaction_counter_pure", False),
    ("impression_counter_min_before_vs_item", False),
    ("top_7_impression_counter_mean_first_3_vs_item", False),
    ("interaction_counter_vs_impression_counter_max_before", False),
    ("price_rem", False),
    ("are_price_sorted", False),
    ("are_price_sorted_rev", False),
    ("prices_sorted_until", False),
    ("prices_sorted_until_current_rank", False),
    ("wrong_price_sorting", False),
    ("clickout_uniq_interactions", False),
    ("clickout_item_uniq_prob", True),
    ("interact_uniq_interactions", False),
    ("interact_item_uniq_prob", True),
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
    "user_id_1cat",
    "cat_action_index_0",
    "cat_action_index_0_norm",
    "cat_action_index_1",
    "cat_action_index_1_norm",
    "cat_action_index_2",
    "cat_action_index_2_norm",
    "cat_action_index_3",
    "cat_action_index_3_norm",
    "cat_action_index_4",
    "cat_action_index_4_norm",
    "cat_action_index_5",
    "cat_action_index_5_norm",
    "cat_action_index_6",
    "cat_action_index_6_norm",
    "cat_action_index_7",
    "cat_action_index_7_norm",
    "cat_action_index_8",
    "cat_action_index_8_norm",
    "cat_action_index_9",
    "cat_action_index_9_norm",
]
numerical_features_offset_2 = ["was_interaction_info", "was_interaction_img", "last_index_diff_5"]


def identity(x):
    return x


def fillna_with_unk(x):
    return "UNK" if x != x else x


def split_by_pipe(x):
    return x.split("|")


def make_vectorizer_1(
    categorical_features=categorical_features_py,
    numerical_features=numerical_features_py,
    numerical_features_offset_2=numerical_features_offset_2,
    numerical_features_for_ranking=numerical_features_for_ranking_py,
):
    return make_pipeline(
        FeatureEng(),
        ColumnTransformer(
            [
                (
                    "numerical",
                    make_pipeline(PandasToNpArray(), SimpleImputer(strategy="constant", fill_value=-9999)),
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
                (
                    "categorical",
                    make_pipeline(PandasToRecords(), DictVectorizer(), SparsityFilter(min_nnz=5)),
                    categorical_features,
                ),
                (
                    "numerical_ranking",
                    make_pipeline(RankFeatures(ascending=False), MinimizeNNZ()),
                    numerical_features_for_ranking + ["clickout_id"],
                ),
                (
                    "numerical_ranking_rev",
                    make_pipeline(RankFeatures(ascending=True), MinimizeNNZ()),
                    numerical_features_for_ranking + ["clickout_id"],
                ),
                ("properties", CountVectorizer(tokenizer=identity, lowercase=False, min_df=2), "properties"),
                (
                    "current_filters",
                    CountVectorizer(preprocessor=fillna_with_unk, tokenizer=split_by_pipe, min_df=2),
                    "current_filters",
                ),
                (
                    "alltime_filters",
                    CountVectorizer(preprocessor=fillna_with_unk, tokenizer=split_by_pipe, min_df=2),
                    "alltime_filters",
                ),
                ("last_10_actions", CountVectorizer(ngram_range=(3, 3), tokenizer=list, min_df=2), "last_10_actions"),
                ("last_poi_bow", CountVectorizer(min_df=5), "last_poi"),
                ("last_event_ts_dict", DictVectorizer(), "last_event_ts_dict"),
                ("actions_tracker", DictVectorizer(), "actions_tracker"),
                (
                    "absolute_rank_0_norm",
                    FeaturesAtAbsoluteRank(rank=0, normalize=True),
                    ["price_vs_mean_price", "rank", "clickout_id"],
                ),
            ]
        ),
    )


def make_vectorizer_2(
    categorical_features=categorical_features_py,
    numerical_features=numerical_features_py,
    numerical_features_offset_2=numerical_features_offset_2,
    numerical_features_for_ranking=numerical_features_for_ranking_py,
):
    return make_pipeline(
        FeatureEng(),
        ColumnTransformer(
            [
                (
                    "numerical",
                    make_pipeline(
                        PandasToNpArray(), SimpleImputer(strategy="constant", fill_value=0), StandardScaler()
                    ),
                    numerical_features + ps_features,
                ),
                (
                    "numerical_context",
                    make_pipeline(
                        LagNumericalFeaturesWithinGroup(),
                        SimpleImputer(strategy="constant", fill_value=0),
                        StandardScaler(),
                    ),
                    numerical_features + ["clickout_id"],
                ),
                (
                    "numerical_context_offset_2",
                    make_pipeline(LagNumericalFeaturesWithinGroup(offset=2), StandardScaler()),
                    numerical_features_offset_2 + ["clickout_id"],
                ),
                ("categorical", make_pipeline(PandasToRecords(), DictVectorizer()), categorical_features),
                ("properties", TfidfVectorizer(tokenizer=identity, lowercase=False, min_df=2), "properties"),
                (
                    "numerical_ranking",
                    make_pipeline(RankFeatures(), StandardScaler()),
                    numerical_features_for_ranking + ["clickout_id"],
                ),
            ]
        ),
        SanitizeSparseMatrix(),
    )


def make_vectorizer_3(
    numerical_features=numerical_features_py, numerical_features_for_ranking=numerical_features_for_ranking_py
):
    return make_pipeline(
        FeatureEng(),
        ColumnTransformer(
            [
                (
                    "numerical",
                    make_pipeline(PandasToNpArray(), SimpleImputer(strategy="constant", fill_value=-9999)),
                    numerical_features,
                ),
                (
                    "numerical_ranking",
                    make_pipeline(RankFeatures(ascending=False), MinimizeNNZ()),
                    numerical_features_for_ranking + ["clickout_id"],
                ),
                (
                    "numerical_ranking_rev",
                    make_pipeline(RankFeatures(ascending=True), MinimizeNNZ()),
                    numerical_features_for_ranking + ["clickout_id"],
                ),
            ]
        ),
    )


def make_vectorizer_3_no_eng(numerical_features, numerical_features_for_ranking):
    return make_pipeline(
        ColumnTransformer(
            [
                (
                    "numerical",
                    make_pipeline(PandasToNpArray(), SimpleImputer(strategy="constant", fill_value=-9999)),
                    numerical_features,
                ),
                (
                    "numerical_ranking",
                    make_pipeline(RankFeatures(ascending=False), MinimizeNNZ()),
                    numerical_features_for_ranking + ["clickout_id"],
                ),
                (
                    "numerical_ranking_rev",
                    make_pipeline(RankFeatures(ascending=True), MinimizeNNZ()),
                    numerical_features_for_ranking + ["clickout_id"],
                ),
                ("actions_tracker", DictVectorizer(), "actions_tracker"),
            ]
        )
    )


def make_vectorizer_4(numerical_features, numerical_features_for_ranking):
    return make_pipeline(
        ColumnTransformer(
            [
                (
                    "numerical",
                    make_pipeline(PandasToNpArray(), SimpleImputer(strategy="constant", fill_value=-9999)),
                    numerical_features,
                ),
                (
                    "numerical_ranking",
                    make_pipeline(RankFeatures(ascending=False), MinimizeNNZ()),
                    numerical_features_for_ranking + ["clickout_id"],
                ),
                (
                    "numerical_ranking_rev",
                    make_pipeline(RankFeatures(ascending=True), MinimizeNNZ()),
                    numerical_features_for_ranking + ["clickout_id"],
                )
            ]
        )
    )


class VectorizeChunks:
    def __init__(self, vectorizer, input_files, output_folder, join_only=False, n_jobs=-2):
        self.vectorizer = vectorizer
        self.input_files = input_files
        self.output_folder = output_folder
        self.join_only = join_only
        self.n_jobs = n_jobs

    def vectorize_all(self):
        # fit vectorizers using the last chunk (I guess the test distribution is more important than training)
        if not self.join_only:
            df = pd.read_csv(sorted(glob.glob(self.input_files))[-1])
            self.vectorizer = self.vectorizer()
            self.vectorizer.fit(df)
        filenames = Parallel(n_jobs=self.n_jobs)(
            delayed(self.vectorize_one)(fn) for fn in sorted(glob.glob(self.input_files))
        )
        metadata_fns, csr_fns = list(zip(*filenames))
        self.save_to_one_file_metadata(metadata_fns)
        self.save_to_one_flie_csrs(csr_fns)

    def save_to_one_file_metadata(self, fns):
        dfs = [pd.read_hdf(os.path.join(self.output_folder, "chunks", fn), key="data") for fn in fns]
        df = pd.concat(dfs, axis=0)
        df.to_hdf(os.path.join(self.output_folder, "meta.h5"), key="data", mode="w")
        gc.collect()

    def save_to_one_flie_csrs(self, fns):
        save_as = os.path.join(self.output_folder, "Xcsr.h5")
        os.unlink(save_as)
        h5f = h5sparse.File(save_as)
        first = True
        for fn in fns:
            logger.info(f"Saving {fn}")
            mat = load_npz(os.path.join(self.output_folder, "chunks", fn)).astype(np.float32)
            if first:
                h5f.create_dataset("matrix", data=mat, chunks=(10_000_000,), maxshape=(None,))
                first = False
            else:
                h5f["matrix"].append(mat)
            gc.collect()
        h5f.close()

    def vectorize_one(self, fn):
        logger.info(f"Vectorize {fn}")
        fname_h5 = fn.split("/")[-1].replace(".csv", ".h5")
        fname_npz = fn.split("/")[-1].replace(".csv", ".npz")
        metadata_save_as = os.path.join(self.output_folder, "chunks", fname_h5)
        sparse_matrix_save_as = os.path.join(self.output_folder, "chunks", fname_npz)

        if self.join_only or os.path.exists(metadata_save_as):
            return (fname_h5, fname_npz)

        df = pd.read_csv(fn)
        mat = self.vectorizer.transform(df)

        df[
            [
                "user_id",
                "session_id",
                "platform",
                "device",
                "city",
                "timestamp",
                "step",
                "clickout_id",
                "item_id",
                "src",
                "is_test",
                "is_val",
                "was_clicked",
            ]
        ].to_hdf(metadata_save_as, key="data", mode="w")

        save_npz(sparse_matrix_save_as, mat)

        gc.collect()

        return (fname_h5, fname_npz)
