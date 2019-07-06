
#include <vector>
#include <string>

const std::vector<std::string> kFeatures{
    "pure",
    "mean",
    "mean_vs_item",
    "min",
    "min_vs_item",
    "max",
    "max_vs_item",
    "rank",
    "rank_norm",
    "mean_before",
    "mean_before_vs_item",
    "min_before",
    "min_before_vs_item",
    "max_before",
    "max_before_vs_item",
    "mean_after",
    "mean_after_vs_item",
    "min_after",
    "min_after_vs_item",
    "max_after",
    "max_after_vs_item",
    "rank_before",
    "rank_norm_before",
    "rank_after",
    "rank_norm_after",
    "min_first_3_vs_item",
    "max_first_3_vs_item",
    "mean_first_3_vs_item",
    "prev_3_vs_item",
    "prev_2_vs_item",
    "prev_1_vs_item",
    "next_1_vs_item",
    "next_2_vs_item",
    "min_prev_3_vs_item",
    "max_prev_3_vs_item",
    "mean_prev_3_vs_item"
};

std::string impressions_num(int item_index, const std::vector<int> &impressions);
std::string impressions_pos(int item_index, const std::vector<int> &impressions);
std::string impressions_pos_norm(int item_index, const std::vector<int> &impressions);

std::string item_price(int item_index, const std::vector<int> &prices);

std::string mean_price(int item_index, const std::vector<int> &prices);
std::string min_price(int item_index, const std::vector<int> &v);
std::string max_price(int item_index, const std::vector<int> &v);

std::string mean_price_vs_item_price(int item_index, const std::vector<int> &prices);
std::string mean_price_vs_item_price(int item_index, const std::vector<int> &prices, std::vector<int> &agg_v);

std::string min_price_vs_item_price(int item_index, const std::vector<int> &v);
std::string min_price_vs_item_price(int item_index, const std::vector<int> &v, std::vector<int> &agg_v);

std::string max_price_vs_item_price(int item_index, const std::vector<int> &v);
std::string max_price_vs_item_price(int item_index, const std::vector<int> &v, std::vector<int> &agg_v);

std::string price_rank(int sorted_item_index, const std::vector<int> &sorted_prices);
std::string price_rank_norm(int sorted_item_index, const std::vector<int> &sorted_prices);
std::string price_vs_price(int item_index, const std::vector<int> &prices, int neighbor);
std::map<std::string, int> get_field_mapping(const std::string &header, const std::vector<int> &indices);
void generate_features(
    int item_index, 
    const std::vector<int> &prices, 
    std::vector<std::string> &row);
