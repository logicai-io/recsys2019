
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

typedef std::unordered_map<int, int> counter_t;

void write_row(const std::vector<std::string> &row, std::ofstream &stream, char separator=',') {
    for (int i = 0; i < row.size()-1; ++i) {
        stream << row[i] << separator;
    }

    stream << row[row.size()-1] << std::endl;
}

const std::string kPath = "/Users/sink/projects/recsys2019/data";
const std::string kRawFilename = "events_sorted.csv";
const std::string kOutputFilename = "elo.csv";
const std::string kClickout = "clickout item";

const int kEloInitValue = 1500;

void find_indices(const std::string &line, char c, std::vector<int> &indices) {
    for (int i = 0; i < line.size(); ++i) {
        if (line[i] == c) {
            indices.push_back(i);
        }
    }
}

int find_index(int item, std::vector<int> &values) {
    for (int i = 0; i < values.size(); ++i) {
        if (values[i] == item) {
            return i;
            break;
        }
    }
    return -1;
}

void parse_list(const std::string &field, std::vector<int> &values) {
    int acc = 0;
    for (int i = 0; i < field.size(); ++i) {
        if (field[i] == '|') {
            values.push_back(acc);
            acc = 0;
        } else {
            acc *= 10;
            acc += field[i] - '0';
        }
    }

    values.push_back(acc);
}

std::string get_action_string(const std::string &line, const std::vector<int> &indices) {
    int offset = 4;
    return line.substr(indices[offset-1]+1, indices[offset] - indices[offset-1] - 1);
}

std::string get_reference_string(const std::string &line, const std::vector<int> &indices) {
    int offset = 5;
    return line.substr(indices[offset-1]+1, indices[offset] - indices[offset-1] - 1);
}

std::string get_impessions_string(const std::string &line, const std::vector<int> &indices) {
    int offset = 7;
    return line.substr(indices[indices.size()-offset]+1, indices[indices.size()-offset+1] - indices[indices.size()-offset] - 1);
}

void init_elo_counter(int item_id, counter_t &counter) {
    if (counter.find(item_id) == counter.end()) {
        counter[item_id] = kEloInitValue;
    }
}

int calc_elo(int player_elo, int opponent_elo, float score) {
    float exp = 1 / (1 + pow(10, ((player_elo - opponent_elo) / 400)));
    return round(player_elo + 32 * (score - exp));
}

void register_win(int winner, int loser, counter_t &counter) {
    int winner_elo = counter[winner];
    int loser_elo = counter[loser];
    counter[winner] = calc_elo(winner_elo, loser_elo, 1.0);
    counter[loser] = calc_elo(loser_elo, winner_elo, 0.0);
}

void update_counters(int selected_index, int item_index, const std::vector<int> &impressions, counter_t &counter) {
    int item_id = impressions[item_index];

    if ((selected_index == item_index) && (item_index != 0)) {
        int previous_id = impressions[item_index-1];
        init_elo_counter(item_id, counter);
        init_elo_counter(previous_id, counter);
        register_win(item_id, previous_id, counter);
    }
}

int main() {
    const auto inputPath = kPath + "/" + "/" + kRawFilename;
    const auto outputPath = kPath + "/" + "/" + kOutputFilename;
    std::ifstream infile(inputPath);
    std::ofstream outfile(outputPath);

    std::string line;

    std::vector<std::string> features{
        "elo_socre"
    };

    //TODO: add header

    write_row(features, outfile);
    std::getline(infile, line); //skip header

    counter_t counter;
    int line_counter = 0;
    int broken_row = 0;

    while (std::getline(infile, line)) {
        if (line_counter % 1000000 == 0) {
            std::cout << line_counter << std::endl;
        }
        line_counter++;
        //user_id,session_id,timestamp,step,action_type,reference,platform,city,device,current_filters,impressions,prices,src,is_test,is_val,fake_impressions,fake_prices
        
        std::vector<int> comma_indices;
        std::vector<int> impressions;

        find_indices(line, ',', comma_indices);

        std::string action_string = get_action_string(line, comma_indices);

        if (get_action_string(line, comma_indices) != kClickout) {
           continue; 
        }

        parse_list(get_impessions_string(line, comma_indices), impressions);
        std::string reference_string = get_reference_string(line, comma_indices);

        int selected_index = find_index(std::stoi(reference_string), impressions);

        /*
        std::cout << reference_string << std::endl;
        std::cout << selected_index << std::endl;
        std::cout << get_impessions_string(line, comma_indices) << std::endl;
        */
        if (selected_index == -1) {
            broken_row++;
            continue;
        }

        std::vector<std::string> row;
        for (int i = 0; i < impressions.size(); ++i) {
            init_elo_counter(impressions[i], counter);
            row.push_back(std::to_string(counter[impressions[i]]));
        }

        write_row(row, outfile, '|');

        for (int i = 0; i < impressions.size(); ++i) {
            update_counters(selected_index, i, impressions, counter);
        }
    }
    std::cout << broken_row << std::endl;

    outfile.close();

    return 0;
}
