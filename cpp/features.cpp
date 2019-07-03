
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

#include "utils.h"


std::string impressions_num(int item_index, const std::vector<int> &impressions) {
    return std::to_string(impressions.size());
}

std::string impressions_pos(int item_index, const std::vector<int> &impressions) {
    return std::to_string(item_index);
}

std::string impressions_pos_norm(int item_index, const std::vector<int> &impressions) {
    return std::to_string(x100(((float)item_index)/impressions.size()));
}

std::string item_price(int item_index, const std::vector<int> &prices) {
    return std::to_string(prices[item_index]);
}

std::string mean_price(int item_index, const std::vector<int> &prices) {
    return std::to_string(x100(std::accumulate(prices.begin(), prices.end(), 0.0)/prices.size()));
}

std::string min_price(int item_index, const std::vector<int> &v) {
    return std::to_string(*std::min_element(v.begin(), v.end()));
}

std::string max_price(int item_index, const std::vector<int> &v) {
    return std::to_string(*std::max_element(v.begin(), v.end()));
}

std::string mean_price_vs_item_price(int item_index, const std::vector<int> &prices) {
    int value = prices[item_index];
    if (value == 0) {
        return std::to_string(10000);
    }

    return std::to_string(x100(std::accumulate(prices.begin(), prices.end(), 0.0)/prices.size() / prices[item_index]));
}

std::string mean_price_vs_item_price(int item_index, const std::vector<int> &prices, std::vector<int> &agg_v) {
    int value = prices[item_index];
    if (value == 0) {
        return std::to_string(10000);
    }
    return std::to_string(x100(std::accumulate(agg_v.begin(), agg_v.end(), 0.0)/agg_v.size() / prices[item_index]));
}

std::string min_price_vs_item_price(int item_index, const std::vector<int> &v) {
    int value = v[item_index];
    if (value == 0) {
        return std::to_string(10000);
    }
    return std::to_string(x100(((float)*std::min_element(v.begin(), v.end())) / v[item_index]));
}

std::string min_price_vs_item_price(int item_index, const std::vector<int> &v, std::vector<int> &agg_v) {
    int value = v[item_index];
    if (value == 0) {
        return std::to_string(10000);
    }
    return std::to_string(x100(((float)*std::min_element(agg_v.begin(), agg_v.end())) / v[item_index]));
}

std::string max_price_vs_item_price(int item_index, const std::vector<int> &v) {
    int value = v[item_index];
    if (value == 0) {
        return std::to_string(10000);
    }
    return std::to_string(x100(((float)*std::max_element(v.begin(), v.end())) / v[item_index]));
}

std::string max_price_vs_item_price(int item_index, const std::vector<int> &v, std::vector<int> &agg_v) {
    int value = v[item_index];
    if (value == 0) {
        return std::to_string(10000);
    }
    return std::to_string(x100(((float)*std::max_element(agg_v.begin(), agg_v.end())) / v[item_index]));
}

std::string price_rank(int sorted_item_index, const std::vector<int> &sorted_prices) {
    return std::to_string(sorted_item_index);
}

std::string price_rank_norm(int sorted_item_index, const std::vector<int> &sorted_prices) {
    return std::to_string(x100(((float)sorted_item_index)/sorted_prices.size()));
}

std::string price_vs_price(int item_index, const std::vector<int> &prices, int neighbor) {
    int neighbor_index = item_index + neighbor;
    neighbor_index = std::min<int>(neighbor_index, prices.size()-1);
    neighbor_index = std::max<int>(neighbor_index, 0);

    if (prices[neighbor_index] == 0) {
        return std::to_string(10000);
    }

    return std::to_string(x100(((float)prices[item_index])/prices[neighbor_index]));
}

void generate_features(
    int item_index, 
    const std::vector<int> &prices, 
    std::vector<std::string> &row) {

    /////////
    // all //
    /////////
    row.push_back(item_price(item_index, prices));
    row.push_back(mean_price(item_index, prices));
    row.push_back(mean_price_vs_item_price(item_index, prices));
    row.push_back(min_price(item_index, prices));
    row.push_back(min_price_vs_item_price(item_index, prices));
    row.push_back(max_price(item_index, prices));
    row.push_back(max_price_vs_item_price(item_index, prices));
    
    //////////
    // rank //
    //////////
    std::vector<int> sorted_prices = prices;
    sort(sorted_prices.begin(), sorted_prices.end());
    int sorted_item_index = find_index(prices[item_index], sorted_prices);

    row.push_back(price_rank(sorted_item_index, sorted_prices));
    row.push_back(price_rank_norm(sorted_item_index, sorted_prices));
    
    ///////////////////
    // values before //
    ///////////////////
    std::vector<int> prices_before(prices.begin(), prices.begin()+item_index+1);
    row.push_back(mean_price(item_index, prices_before));
    row.push_back(mean_price_vs_item_price(item_index, prices_before));
    row.push_back(min_price(item_index, prices_before));
    row.push_back(min_price_vs_item_price(item_index, prices_before));
    row.push_back(max_price(item_index, prices_before));
    row.push_back(max_price_vs_item_price(item_index, prices_before));

    //////////////////
    // values after //
    //////////////////
    std::vector<int> prices_after(prices.begin()+item_index, prices.end());
    row.push_back(mean_price(0, prices_after));
    row.push_back(mean_price_vs_item_price(0, prices_after));
    row.push_back(min_price(0, prices_after));
    row.push_back(min_price_vs_item_price(0, prices_after));
    row.push_back(max_price(0, prices_after));
    row.push_back(max_price_vs_item_price(0, prices_after));

    /////////////////
    // rank before //
    /////////////////
    std::vector<int> sorted_prices_before = prices_before;
    sort(sorted_prices_before.begin(), sorted_prices_before.end());
    int sorted_item_index_before = find_index(prices[item_index], sorted_prices_before);
    row.push_back(price_rank(sorted_item_index_before, sorted_prices_before));
    row.push_back(price_rank_norm(sorted_item_index_before, sorted_prices_before));

    ////////////////
    // rank after //
    ////////////////
    std::vector<int> sorted_prices_after = prices_after;
    sort(sorted_prices_after.begin(), sorted_prices_after.end());
    int sorted_item_index_after = find_index(prices[item_index], sorted_prices_after);
    row.push_back(price_rank(sorted_item_index_after, sorted_prices_after));
    row.push_back(price_rank_norm(sorted_item_index_after, sorted_prices_after));

    /////////////
    // first 3 //
    /////////////
    std::vector<int> first3(prices.begin(), prices.begin()+std::min<int>(3, prices.size()));
    row.push_back(min_price_vs_item_price(item_index, prices, first3));
    row.push_back(max_price_vs_item_price(item_index, prices, first3));
    row.push_back(mean_price_vs_item_price(item_index, prices, first3));
    
    ///////////////
    // neighbors //
    ///////////////
    row.push_back(price_vs_price(item_index, prices, -3));
    row.push_back(price_vs_price(item_index, prices, -2));
    row.push_back(price_vs_price(item_index, prices, -1));
    row.push_back(price_vs_price(item_index, prices, 1));
    row.push_back(price_vs_price(item_index, prices, 2));
    
    ////////////////
    // previous 3 //
    ////////////////
    int offset = std::max(item_index-3, 0);
    std::vector<int> previous3(prices.begin()+offset, prices.begin()+item_index+1);
    row.push_back(min_price_vs_item_price(item_index, prices, previous3));
    row.push_back(max_price_vs_item_price(item_index, prices, previous3));
    row.push_back(mean_price_vs_item_price(item_index, prices, previous3));
}
