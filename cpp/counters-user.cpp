#include <string>
#include <unordered_map>
#include <vector>

#include "counters-user.h"
#include <algorithm>
#include <numeric>
#include "utils.h"

int mean(const std::vector<int> vals) {
    return x100(std::accumulate(vals.begin(), vals.end(), 0.0)/vals.size());
}

int min(const std::vector<int> vals) {
    return *std::min_element(vals.begin(), vals.end());
}

int max(const std::vector<int> vals) {
    return *std::max_element(vals.begin(), vals.end());
}

int mean(const std::vector<int> vals, int limit) {
    int size = std::min((int)vals.size(), limit);
    return x100(std::accumulate(vals.begin(), vals.begin()+size, 0.0)/size);
}

int min(const std::vector<int> vals, int limit) {
    int size = std::min((int)vals.size(), limit);
    return *std::min_element(vals.begin(), vals.begin() + size);
}

int max(const std::vector<int> vals, int limit) {
    int size = std::min((int)vals.size(), limit);
    return *std::max_element(vals.begin(), vals.begin() + size);
}

std::string SessionCounter::getName() const {
    return std::string("diff_num,")
        + "rank_diffs_mean,time_diff_mean,time_vs_rank_mean,"
        + "rank_diffs_min,time_diff_min,time_vs_rank_min,"
        + "rank_diffs_max,time_diff_max,time_vs_rank_max,"
        + "rank_diffs_mean_5,time_diff_mean_5,time_vs_rank_mean_5,"
        + "rank_diffs_min_5,time_diff_min_5,time_vs_rank_min_5,"
        + "rank_diffs_max_5,time_diff_max_5,time_vs_rank_max_5,"
        + "estimated_move";
};

std::string to_str(const std::vector<int> &vals) {
    //TODO: make efficient
    std::string result = "";
    for (int i = 0; i < vals.size(); ++i) {
        result += std::to_string(vals[i]);
        if (i != vals.size()-1) {
            result += ",";
        }
    }

    return result;
};

std::string SessionCounter::getFeatures(const std::string &user_id, const std::string &timestamp) {

    std::vector<int> features;
    int timestamp_int = std::stoi(timestamp);

    if (rank_diffs[user_id].size() == 0) {
        return std::string("0,")
            + "0,0,0,"
            + "0,0,0,"
            + "0,0,0,"
            + "0,0,0,"
            + "0,0,0,"
            + "0,0,0,"
            + "0";
    }

    features.push_back(time_diffs[user_id].size());

    features.push_back(mean(rank_diffs[user_id]));
    features.push_back(mean(time_diffs[user_id]));
    features.push_back(mean(time_vs_ranks[user_id]));

    features.push_back(min(rank_diffs[user_id]));
    features.push_back(min(time_diffs[user_id]));
    features.push_back(min(time_vs_ranks[user_id]));

    features.push_back(max(rank_diffs[user_id]));
    features.push_back(max(time_diffs[user_id]));
    features.push_back(max(time_vs_ranks[user_id]));

    features.push_back(mean(rank_diffs[user_id], 5));
    features.push_back(mean(time_diffs[user_id], 5));
    features.push_back(mean(time_vs_ranks[user_id], 5));

    features.push_back(min(rank_diffs[user_id], 5));
    features.push_back(min(time_diffs[user_id], 5));
    features.push_back(min(time_vs_ranks[user_id], 5));

    features.push_back(max(rank_diffs[user_id], 5));
    features.push_back(max(time_diffs[user_id], 5));
    features.push_back(max(time_vs_ranks[user_id], 5));

    float mean_speed = std::accumulate(time_vs_ranks[user_id].begin(), time_vs_ranks[user_id].end(), 0.0)/time_vs_ranks[user_id].size();
    int estimated_move = x100(((float)timestamp_int - last_timestamp[user_id])/mean_speed);
    features.push_back(estimated_move);

    return to_str(features);
};



void SessionCounter::update(
    const std::string &user_id, 
    const std::string &session_id, 
    const std::string &timestamp, 
    const std::string &action_type, 
    const std::string &reference, 
    const std::vector<int> &impressions) {

    int timestamp_int;
    int reference_int;
    try {
        timestamp_int = std::stoi(timestamp);
        reference_int = std::stoi(reference);
    } catch (std::exception e) {
        return;
    }

    if (action_type != "interaction item image" &&
        action_type != "clickout item" &&
        action_type != "interaction item info" &&
        action_type != "interaction item rating" &&
        action_type != "interaction item deals") {
        return;

    }

    auto it = std::find(impressions.begin(), impressions.end(), reference_int);
    if (it == impressions.end()) {
        return;
    }
    int index = std::distance(impressions.begin(), it);;
    std::string impressions_str = to_str(impressions);

    if (last_impressions[user_id] == impressions_str && 
        last_session_id[user_id] == session_id &&
        last_index[user_id] != index) {

        int rank_diff = std::abs(last_index[user_id] - index);
        int time_diff = timestamp_int - last_timestamp[user_id];
        int time_vs_rank = x100(((float)time_diff) / rank_diff);

        rank_diffs[user_id].push_back(rank_diff);
        time_diffs[user_id].push_back(time_diff);
        time_vs_ranks[user_id].push_back(time_vs_rank);

    }

    last_timestamp[user_id] = timestamp_int;
    last_index[user_id] = index;
    last_action_type[user_id] = action_type;
    last_session_id[user_id] = session_id;
    last_impressions[user_id] = impressions_str;
};

