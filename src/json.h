// Alpacca — minimal JSON value, parser and writer.
// Just enough for registry manifests and Alpacca's own model manifests;
// it is NOT in any inference path. MIT License. See LICENSE.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace alpacca {

struct JValue {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ };

    Type        type = NUL;
    bool        b    = false;
    double      num  = 0.0;
    std::string str;
    std::vector<JValue> arr;
    std::vector<std::pair<std::string, JValue>> obj; // insertion-ordered

    static JValue make_str(const std::string & s);
    static JValue make_num(double n);
    static JValue make_bool(bool v);
    static JValue make_obj();
    static JValue make_arr();

    bool is_obj() const { return type == OBJ; }
    bool is_arr() const { return type == ARR; }
    bool is_str() const { return type == STR; }
    bool is_num() const { return type == NUM; }

    // Object member lookup; nullptr when absent or not an object.
    const JValue * find(const std::string & key) const;
    // String member with default.
    std::string get_str(const std::string & key, const std::string & def = "") const;
    // Numeric member with default (also accepts numeric strings).
    double get_num(const std::string & key, double def = 0.0) const;

    void set(const std::string & key, JValue v); // add/replace object member

    // Serialize. indent < 0 → compact; otherwise pretty with given indent.
    std::string dump(int indent = -1) const;
};

// Parse JSON text. On failure returns false and sets *err.
bool json_parse(const std::string & text, JValue & out, std::string & err);

} // namespace alpacca
