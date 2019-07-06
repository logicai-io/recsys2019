
#include <string>
#include <unordered_map>
#include <vector>

const std::string kClickoutItem("clickout item");

typedef std::unordered_map<int, int> counter_t;

class Counter {
   public:
      virtual std::string getName() const = 0;
      virtual int get(int identifier) const = 0;
      virtual void update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions) = 0;
      virtual ~Counter() {};
};

class ImpressionCounter : public Counter {
    counter_t counter;
  public:
    std::string getName() const;
    int get(int identifier) const;
    void update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions);
};

class VerifiedImpressionCounter : public Counter {
    counter_t counter;
  public:
    std::string getName() const;
    int get(int identifier) const;
    void update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions);
};

class SpecificInteractionCounter : public Counter {
    counter_t counter;
    std::string action_type;
  public:
    SpecificInteractionCounter(std::string action_type) : action_type(action_type) {};
    std::string getName() const;
    int get(int identifier) const;
    void update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions);
};

class InteractionCounter : public Counter {
    counter_t counter;
  public:
    std::string getName() const;
    int get(int identifier) const;
    void update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions);
};

class ClickoutCounter : public Counter {
    counter_t counter;
  public:
    std::string getName() const;
    int get(int identifier) const;
    void update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions);
};

class MeanRankCounter : public Counter {
    std::unordered_map<int, long> rank_sum_counter;
    counter_t impressions_counter;
  public:
    std::string getName() const;
    int get(int identifier) const;
    void update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions);
};

class ConditionalMeanRankCounter : public Counter {
    std::unordered_map<int, long> rank_sum_counter;
    counter_t impressions_counter;
  public:
    std::string getName() const;
    int get(int identifier) const;
    void update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions);
};

class TopImpressionCounter : public Counter {
    int top;
    counter_t counter;
  public:
    TopImpressionCounter(int top) : top(top) {};
    std::string getName() const;
    int get(int identifier) const;
    void update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions);
};

class ComparisonCounter : public Counter {
    Counter* counter_a;
    Counter* counter_b;
  public:
    ComparisonCounter(Counter* counter_a, Counter* counter_b) : counter_a(counter_a), counter_b(counter_b) {};
    std::string getName() const;
    int get(int identifier) const;
    void update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions);
    ~ComparisonCounter();
};

class IdentifierCounter: public Counter {
  public:
    std::string getName() const;
    int get(int identifier) const;
    void update(const std::string &actionType, const std::string &reference, const std::vector<int> &impressions);
};


std::string map_over_counter(const Counter &counter, const std::vector<int> &impressions);
