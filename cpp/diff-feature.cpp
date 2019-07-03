
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "counters-user.h"
#include "utils.h"
const std::string kClickoutItem("clickout item");

const std::string kInputPath = "/Users/sink/projects/recsys2019/data/events_sorted.csv";
const std::string kOutputPath = "/Users/sink/projects/recsys2019/data/diff-feature.csv";

const int kLineLimit = 1000000;

int main() {
    std::string line;
    std::vector<std::string> features;


    SessionCounter counter;

    std::ifstream infile(kInputPath);
    std::ofstream outfile(kOutputPath);

    features.push_back(counter.getName());

    // handle header
    write_row(features, outfile);
    std::getline(infile, line);
    std::map<std::string, int> mapping = get_field_mapping(line);

    // transform features
    int line_counter = 0;
    while (std::getline(infile, line)) {

        std::vector<int> indices;
        std::vector<int> impressions;
        std::vector<std::string> row;

        if (update_counter(line_counter, kLineLimit)) {
            break;
        }

        // parse
        find_indices(line, ',', indices);

        std::string impressions_string = get_field(mapping["fake_impressions"], line, indices);
        std::string reference = get_field(mapping["reference"], line, indices);
        std::string action_type = get_field(mapping["action_type"], line, indices);
        std::string timestamp = get_field(mapping["timestamp"], line, indices);
        std::string user_id = get_field(mapping["user_id"], line, indices);
        std::string session_id = get_field(mapping["session_id"], line, indices);

        parse_list(impressions_string, impressions);

        // generate features
        row.push_back(counter.getFeatures(user_id, timestamp));


        // update counters
        counter.update(user_id, session_id, timestamp, action_type, reference, impressions);

        if (get_field(mapping["action_type"], line, indices) != kClickoutItem) {
            continue;
        }

        for (int i : impressions) {
            write_row(row, outfile, ',');
        }
    }

    // cleanup
    outfile.close();


    return 0;
}
