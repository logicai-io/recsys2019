
#include <map>
#include <vector>
#include <string>

void print(const std::string &message);
void print(int message);

void parse_list(const std::string &field, std::vector<int> &values);

void write_row(const std::vector<std::string> &row, std::ofstream &stream);
void write_row(const std::vector<std::string> &row, std::ofstream &stream, char c);

void find_indices(const std::string &line, char c, std::vector<int> &indices);
std::vector<int> find_indices(const std::string &line, char c);
int find_index(int item, std::vector<int> &values);


std::string get_field(const int field_idx, const std::string &line, const std::vector<int> &indices);
std::map<std::string, int> get_field_mapping(const std::string &header);


bool update_counter(int& line_counter, int limit);
int x100(float x);
int x10000(float x);
