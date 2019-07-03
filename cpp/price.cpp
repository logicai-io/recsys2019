
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <map>

#include "features.h"
#include "utils.h"

const std::string kInputPath = "../data/events_sorted.csv";
const std::string kOutputPath = "../data/price.csv";

const std::string kClickoutItem("clickout item");
const int kLineLimit = 50000000;


const std::string kName = "price";

bool update_counter(int& line_counter) {
    if (line_counter % 1000000 == 0) {
        std::cout << line_counter << std::endl;
    }

    line_counter++;

    return line_counter > kLineLimit;
        
}

void generate_features(
    int item_index, 
    const std::vector<int> &impressions, 
    const std::vector<int> &prices, 
    std::vector<std::string> &row) {

    ////////////////
    // all prices //
    ////////////////
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
    // prices before //
    ///////////////////
    std::vector<int> prices_before(prices.begin(), prices.begin()+item_index+1);
    row.push_back(mean_price(item_index, prices_before));
    row.push_back(mean_price_vs_item_price(item_index, prices_before));
    row.push_back(min_price(item_index, prices_before));
    row.push_back(min_price_vs_item_price(item_index, prices_before));
    row.push_back(max_price(item_index, prices_before));
    row.push_back(max_price_vs_item_price(item_index, prices_before));

    //////////////////
    // prices after //
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

int main() {
    std::string line;

    std::ifstream infile(kInputPath);
    std::ofstream outfile(kOutputPath);

    // handle header
    std::vector<std::string> features = kFeatures;
    for (int i = 0; i < features.size(); ++i) {
        features[i] = features[i] + "_" + kName;
    }

    write_row(features, outfile); // create header
    std::getline(infile, line); //skip header
    std::map<std::string, int> mapping = get_field_mapping(line);

    int line_counter = 0;
    while (std::getline(infile, line)) {
        std::vector<int> indices;
        std::vector<int> prices;
        std::vector<int> impressions;

        if (update_counter(line_counter)) {
            break;
        }

        find_indices(line, ',', indices);
        
        // take only clickout items
        if (get_field(mapping["action_type"], line, indices) != kClickoutItem) {
            continue;
        }
        
        parse_list(get_field(mapping["fake_impressions"], line, indices), impressions);
        parse_list(get_field(mapping["fake_prices"], line, indices), prices);

        for (int i : impressions) {
            std::string item_id_string = std::to_string(i);
            std::vector<std::string> row;

            int item_index = find_index(std::stoi(item_id_string), impressions);
            if (item_index == -1) {
                std::cout << "Invalid line" << std::endl;
                continue;
            }

            generate_features(item_index, impressions, prices, row);
            write_row(row, outfile);
        }
    }

    outfile.close();

    return 0;
}
