
#include "counters.h"
#include "utils.h"


std::string map_over_counter(const Counter &counter, const std::vector<int> &impressions) {
    //TODO: make this efficient
    std::string result = "";
    for (int i = 0; i < impressions.size(); ++i) {
        const int value = counter.get(impressions[i]);
        result += std::to_string(value);
        if (i != impressions.size()-1) {
            result += "|";
        }
    }

    return result;
};

std::string ImpressionCounter::getName() const {
    return "impression_counter";
}

void ImpressionCounter::update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions) {
        if (actionType == kClickoutItem) {
            for (int i : impressions) {
                counter[i]++;
            }
    }
}

int ImpressionCounter::get(int identifier) const {
    auto it = counter.find(identifier);
    if (it != counter.end()) {
        return it->second;
    } else {
        return 0;
    }
}

std::string VerifiedImpressionCounter::getName() const {
    return "verified_impression_counter";
}

void VerifiedImpressionCounter::update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions) {
    if (actionType == kClickoutItem) {
        try {
          int reference_id = std::stoi(reference);
          int i = 0;
          bool found = false;
          while (i < impressions.size() && !found) {
            counter[impressions[i]]++;
            if (impressions[i] == reference_id) {
                found = true;
            }
            i++;
          }
        } catch (const std::invalid_argument& ia) {
          //pass
        }
    }
}

int VerifiedImpressionCounter::get(int identifier) const {
    auto it = counter.find(identifier);
    if (it != counter.end()) {
        return it->second;
    } else {
        return 0;
    }
}

std::string SpecificInteractionCounter::getName() const {
    return "specific_impression_counter_" + action_type ;
}

void SpecificInteractionCounter::update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions) {
    if (actionType != action_type) {
        return;
    }

    try {
      //TODO: make this correct - only for interactions
      int reference_id = std::stoi(reference);
      counter[reference_id]++;
    } catch (const std::invalid_argument& ia) {
      //pass
    }
}

int SpecificInteractionCounter::get(int identifier) const {
    auto it = counter.find(identifier);
    if (it != counter.end()) {
        return it->second;
    } else {
        return 0;
    }
}


std::string InteractionCounter::getName() const {
    return "interaction_counter";
}

void InteractionCounter::update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions) {
    try {
      //TODO: make this correct - only for interactions
      int reference_id = std::stoi(reference);
      counter[reference_id]++;
    } catch (const std::invalid_argument& ia) {
      //pass
    }
}

int InteractionCounter::get(int identifier) const {
    auto it = counter.find(identifier);
    if (it != counter.end()) {
        return it->second;
    } else {
        return 0;
    }
}

std::string ClickoutCounter::getName() const {
    return "clickout_counter";
}

void ClickoutCounter::update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions) {
   if (actionType == kClickoutItem) {
       try {
           int reference_id = std::stoi(reference);
           counter[reference_id]++;
       } catch (const std::invalid_argument& ia) {
           //pass
       }
   }
}

int ClickoutCounter::get(int identifier) const {
    auto it = counter.find(identifier);
    if (it != counter.end()) {
        return it->second;
    } else {
        return 0;
    }
}

std::string MeanRankCounter::getName() const {
    return "mean_rank_counter";
}

void MeanRankCounter::update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions) {
    if (actionType == kClickoutItem) {
        for (int i = 0; i < impressions.size(); ++i) {
            int identifier = impressions[i];
            rank_sum_counter[identifier] += i;
            impressions_counter[identifier]++;
        }
    }
}

int MeanRankCounter::get(int identifier) const {
    auto rank_sum_it = rank_sum_counter.find(identifier);
    auto impressions_it = impressions_counter.find(identifier);
    if (rank_sum_it == rank_sum_counter.end() || impressions_it == impressions_counter.end()) {
        return 10000;
    } else {
        return x100((float) rank_sum_it->second / impressions_it->second);
    }
}

std::string ConditionalMeanRankCounter::getName() const {
    return "conditional_mean_rank_counter";
}

void ConditionalMeanRankCounter::update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions) {
    if (actionType == kClickoutItem) {
       try {
          int reference_id = std::stoi(reference);
          for (int i = 0; i < impressions.size(); ++i) {
              int identifier = impressions[i];

              if (reference_id == identifier) {
                rank_sum_counter[identifier] += i;
                impressions_counter[identifier]++;
                return;
              }
          }
       } catch (const std::invalid_argument& ia) {
            //pass
       }
    }
}

int ConditionalMeanRankCounter::get(int identifier) const {
    auto rank_sum_it = rank_sum_counter.find(identifier);
    auto impressions_it = impressions_counter.find(identifier);
    if (rank_sum_it == rank_sum_counter.end() || impressions_it == impressions_counter.end()) {
        return 10000;
    } else {
        return x100((float) rank_sum_it->second / impressions_it->second);
    }
}

std::string TopImpressionCounter::getName() const {
    return "top_" + std::to_string(top) + "_impression_counter";
}

void TopImpressionCounter::update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions) {
    if (actionType == kClickoutItem) {
        for (int i = 0; i < std::min<int>(impressions.size(), top); ++i) {
            int identifier = impressions[i];
            counter[identifier]++;
        }
    }
}

int TopImpressionCounter::get(int identifier) const {
    auto it = counter.find(identifier);
    if (it != counter.end()) {
        return it->second;
    } else {
        return 0;
    }
}

std::string ComparisonCounter::getName() const {
    return counter_a->getName() + "_vs_" + counter_b->getName();
}

void ComparisonCounter::update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions) {
    counter_a->update(actionType, reference, impressions);
    counter_b->update(actionType, reference, impressions);
}

int ComparisonCounter::get(int identifier) const {
    int b = counter_b->get(identifier);
    if (b == 0) {
        return 1000000;
    }
    return x10000((float)counter_a->get(identifier) / b);
}

ComparisonCounter::~ComparisonCounter() {
    delete counter_a;
    delete counter_b;
}

std::string IdentifierCounter::getName() const {
    return "identifier_counter";
}

void IdentifierCounter::update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions) { }

int IdentifierCounter::get(int identifier) const {
    return identifier;
}
