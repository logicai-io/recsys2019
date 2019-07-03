
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "counters.h"
#include "utils.h"

const std::string kInputPath = "../data/events_sorted.csv";
const std::string kOutputPath = "../data/scores.csv";

const int kLineLimit = 20000000;

std::vector<Counter*> counters = {
    new ClickoutCounter(),
    new ImpressionCounter(),
    new InteractionCounter(),
    new MeanRankCounter(),
    new TopImpressionCounter(7),
    new ComparisonCounter(
            new InteractionCounter(),
            new ImpressionCounter()),
    new ComparisonCounter(
            new ClickoutCounter(),
            new ImpressionCounter()),
    new ComparisonCounter(
            new ClickoutCounter(),
            new InteractionCounter()),
    new IdentifierCounter(),
    // verified impressions
    new VerifiedImpressionCounter(),
    new ComparisonCounter(
            new ClickoutCounter(),
            new VerifiedImpressionCounter()),
    new ComparisonCounter(
            new InteractionCounter(),
            new VerifiedImpressionCounter()),
    new ComparisonCounter(
            new VerifiedImpressionCounter(),
            new ImpressionCounter()),

    // specific interactions
    new SpecificInteractionCounter("interaction item info"),
    new SpecificInteractionCounter("interaction item rating"),
    new SpecificInteractionCounter("interaction item deals"),
    new ComparisonCounter(
            new SpecificInteractionCounter("interaction item info"),
            new VerifiedImpressionCounter()),
    new ComparisonCounter(
            new SpecificInteractionCounter("interaction item rating"),
            new VerifiedImpressionCounter()),
    new ComparisonCounter(
            new SpecificInteractionCounter("interaction item deals"),
            new VerifiedImpressionCounter()),
    // conditional mean rank counter
    new ConditionalMeanRankCounter(),

    // top x impressions
    new TopImpressionCounter(3),
    new TopImpressionCounter(5),
    new ComparisonCounter(
            new TopImpressionCounter(3),
            new ImpressionCounter()),
    new ComparisonCounter(
            new TopImpressionCounter(5),
            new ImpressionCounter()),
    new ComparisonCounter(
            new TopImpressionCounter(7),
            new ImpressionCounter()),
};

int main() {
    std::string line;
    std::vector<std::string> features;

    std::ifstream infile(kInputPath);
    std::ofstream outfile(kOutputPath);

    for (auto c : counters) {
        features.push_back(c->getName());
    }

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

        parse_list(impressions_string, impressions);

        // generate features
        for (auto c : counters) {
            row.push_back(map_over_counter(*c, impressions));
        }


        // update counters
        for (auto c : counters) {
            c->update(action_type, reference, impressions);
        }

        if (get_field(mapping["action_type"], line, indices) != kClickoutItem) {
            continue;
        }

        write_row(row, outfile, ',');
    }

    // cleanup
    outfile.close();

    for (Counter* c : counters) {
        delete c;
    }

    return 0;
}
