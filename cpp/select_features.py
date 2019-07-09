import pandas as pd

df = pd.read_csv("../data/comparison.csv")
df = df[["clickout_counter_vs_interaction_counter_mean",
"mean_rank_counter_mean",
"identifier_counter_min_after",
"interaction_counter_pure",
"identifier_counter_max_after",
"identifier_counter_mean_before_vs_item",
"identifier_counter_prev_2_vs_item",
"interaction_counter_max_vs_item",
"interaction_counter_mean",
"mean_rank_counter_mean_after_vs_item",
"mean_rank_counter_rank_norm_after",
"mean_rank_counter_max_vs_item",
"mean_rank_counter_min",
"impression_counter_prev_1_vs_item",
"impression_counter_mean_before_vs_item",
"clickout_counter_vs_impression_counter_max_after",
"clickout_counter_vs_impression_counter_max_before",
"identifier_counter_rank_norm_after",
"impression_counter_rank_norm",
"impression_counter_mean_prev_3_vs_item",
"clickout_counter_vs_interaction_counter_pure",
"impression_counter_min_before_vs_item",
"top_7_impression_counter_mean_first_3_vs_item",
"interaction_counter_vs_impression_counter_max_before"]]

df.to_csv("../data/features/comp_v0_selected.csv", index=False)