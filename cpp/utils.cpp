

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>


void write_row(const std::vector<std::string> &row, std::ofstream &stream, char sep) {
    for (int i = 0; i < row.size()-1; ++i) {
        stream << row[i] << sep;
    }

    stream << row[row.size()-1] << std::endl;
}

void write_row(const std::vector<std::string> &row, std::ofstream &stream) {
    write_row(row, stream, ',');
}

void print(const std::string &message) {
    std::cout << message << std::endl;
}

void print(int message) {
    std::cout << message << std::endl;
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

void find_indices(const std::string &line, char c, std::vector<int> &indices) {
    bool in_quotes = false;
    for (int i = 0; i < line.size(); ++i) {
        if (line[i] == '\"') {
            in_quotes = !in_quotes;
        } else {
          if (line[i] == c && !in_quotes) {
              indices.push_back(i);
          }
        }
    }
}

std::vector<int> find_indices(const std::string &line, char c) {
    std::vector<int> indices;
    find_indices(line, c, indices);
    return indices;
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

std::string get_impessions_string(const std::string &line, const std::vector<int> &indices) {
    int offset = 5;
    return line.substr(indices[indices.size()-offset]+1, indices[indices.size()-offset+1] - indices[indices.size()-offset] - 1);
}

std::string get_prices_string(const std::string &line, const std::vector<int> &indices) {
    int offset = 4;
    return line.substr(indices[indices.size()-offset]+1, indices[indices.size()-offset+1] - indices[indices.size()-offset] - 1);
}

std::string get_action_type_string(const std::string &line, const std::vector<int> &indices) {
    return line.substr(indices[3]+1, indices[4]-indices[3]-1);
}

std::string get_timestamp_string(const std::string &line, const std::vector<int> &indices) {
    int offset = 2;
    return line.substr(indices[offset-1]+1, indices[offset] - indices[offset-1] - 1);
}

std::string get_session_id_string(const std::string &line, const std::vector<int> &indices) {
    int offset = 1;
    return line.substr(indices[offset-1]+1, indices[offset] - indices[offset-1] - 1);
}

std::string get_step_string(const std::string &line, const std::vector<int> &indices) {
    int offset = 3;
    return line.substr(indices[offset-1]+1, indices[offset] - indices[offset-1] - 1);
}


std::string get_reference_string(const std::string &line, const std::vector<int> &indices) {
    int offset = 5;
    return line.substr(indices[offset-1]+1, indices[offset] - indices[offset-1] - 1);
}

std::string get_field(const int field_idx, const std::string &line, const std::vector<int> &indices) {
    if (field_idx == 0) {
        return line.substr(0, indices[0]-1);
    }

    if (field_idx == indices.size()) {
        return line.substr(indices[field_idx-1]+1);
    }

    return line.substr(indices[field_idx-1]+1, indices[field_idx]-indices[field_idx-1]-1);
}

std::map<std::string, int> get_field_mapping(const std::string &header) {
    std::vector<int> indices = find_indices(header, ',');

    std::map<std::string, int> mapping;
    for (int i = 0; i < indices.size()+1; ++i) {
        int modifier = 1;
        if (i == 0) {
            modifier = 0;
        }

        std::string field;
        if (i == indices.size()) {
            field = header.substr(indices[i-1]+modifier);
        } else if (i == 0) {
            field = header.substr(0, indices[0]);
        } else {
            field = header.substr(indices[i-1]+modifier, indices[i] - indices[i-1]-modifier);
        }
        mapping[field] = i;
    }

    return mapping;
}

bool update_counter(int& line_counter, int limit) {
    if (line_counter % 1000000 == 0) {
        std::cout << line_counter << std::endl;
    }

    line_counter++;

    return line_counter > limit;
}

int x100(float x) {
    return (int)(x * 100);
}

int x10000(float x) {
    return (int)(x * 10000);
}
