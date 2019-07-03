
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.h"

const std::string kInputPath = "/Users/sink/projects/recsys2019/data/events_sorted.csv";
const std::string kOutputPath = "/Users/sink/projects/recsys2019/data/events.csv";
const int kLineLimit = 10000;

const std::set<std::string> kActions = {
    "interaction item image",
    "clickout item",
    "search for item",
    "interaction item info",
    "interaction item rating",
    "interaction item deals"
};

const std::vector<std::string> features{
    "session_id",
    "timestamp",
    "step", 
    "index",
    "action",
    "impression_num"
};

int main() {
    std::string line;

    std::ifstream infile(kInputPath);
    std::ofstream outfile(kOutputPath);


    write_row(features, outfile);
    std::getline(infile, line);
    std::map<std::string, int> mapping = get_field_mapping(line);
    
    int line_counter = 0;

    while (std::getline(infile, line)) {

        if (line_counter % 1000000 == 0) {
            std::cout << line_counter << std::endl;
        }

        if (line_counter > kLineLimit) {
            break;
        }

        line_counter++;

        std::vector<int> indices;
        std::vector<int> impressions;
        int selected_index = -1;
        
        find_indices(line, ',', indices);

        std::string action_string = get_field(mapping["action_type"], line, indices);
        if (kActions.find(action_string) == kActions.end()) {
            continue;
        }

        parse_list(get_field(mapping["fake_impressions"], line, indices), impressions);
        try {
            selected_index = find_index(std::stoi(get_field(mapping["reference"], line, indices)), impressions);
        } catch (const std::invalid_argument& ia) {
            selected_index = -1;
        }

        std::vector<std::string> row;
        std::string selected_index_string;
        if (selected_index == -1) {
             selected_index_string = "";
        } else {
            selected_index_string = std::to_string(selected_index);
        }

        row.push_back(get_field(mapping["session_id"], line, indices));
        row.push_back(get_field(mapping["timestamp"], line, indices));
        row.push_back(get_field(mapping["step"], line, indices));
        row.push_back(selected_index_string);
        row.push_back(action_string);
        row.push_back(std::to_string(impressions.size()));

        write_row(row, outfile);
    }

    outfile.close();

    return 0;
}
