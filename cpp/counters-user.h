
#include <string>
#include <unordered_map>
#include <vector>

class SessionCounter {
    std::unordered_map<std::string, int> last_timestamp;
    std::unordered_map<std::string, int> last_index;
    std::unordered_map<std::string, std::string> last_action_type;
    std::unordered_map<std::string, std::string> last_session_id;
    std::unordered_map<std::string, std::string> last_impressions;

    std::unordered_map<std::string, std::vector<int>> rank_diffs;
    std::unordered_map<std::string, std::vector<int>> time_diffs;
    std::unordered_map<std::string, std::vector<int>> time_vs_ranks;

   public:
      std::string getName() const;
      std::string getFeatures(
              const std::string &user_id,
              const std::string &timestamp);

      void update(
              const std::string &user_id, 
              const std::string &session_id, 
              const std::string &timestamp, 
              const std::string &action_type, 
              const std::string &reference, 
              const std::vector<int> &impressions);
      ~SessionCounter() {};
};

