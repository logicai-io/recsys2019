
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <map>

#include "features.h"
#include "utils.h"

const std::string kInputPath = "../data/scores.csv";
const std::string kOutputPath = "../data/comparison.csv";

const int kLineLimit = 50000000;

int main() {
    std::string line;

    std::ifstream infile(kInputPath);
    std::ofstream outfile(kOutputPath);

    std::getline(infile, line); //skip header
    std::map<std::string, int> mapping = get_field_mapping(line);

    std::vector<std::string> features;
    for (auto kv : mapping) {
        std::string base_name = kv.first;
        print(base_name);
        for (auto comp_name : kFeatures) {
            features.push_back(base_name + "_" + comp_name);
            print(base_name + "_" + comp_name);
        }
    }

    write_row(features, outfile); // create header

    int line_counter = 0;
    while (std::getline(infile, line)) {
        std::vector<int> indices;
        std::vector<int> pipes;

        if (update_counter(line_counter, kLineLimit)) {
            break;
        }

        find_indices(line, ',', indices);
        find_indices(line, '|', pipes);

        //parse input line
        std::map<std::string, std::vector<int>> scores;
        for (auto score : mapping) {
            std::vector<int> values;
            std::string column_name = score.first;

            parse_list(get_field(mapping[column_name], line, indices), values);
            scores[column_name] = values;
        }

        //produce output lines
        int impression_num = (pipes.size() / (indices.size() + 1)) + 1;
        for (int i = 0; i < impression_num; ++i) {
            std::vector<std::string> row;
            for (auto score : mapping) {
                generate_features(i, scores[score.first], row);
            }
            write_row(row, outfile);
        }
    }

    outfile.close();

    return 0;
}
