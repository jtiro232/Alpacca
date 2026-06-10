// Alpacca — minimal JSON parser and writer. MIT License. See LICENSE.
#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace alpacca {

// ---- value helpers ------------------------------------------------------

JValue JValue::make_str(const std::string & s) {
    JValue v;
    v.type = STR;
    v.str = s;
    return v;
}

JValue JValue::make_num(double n) {
    JValue v;
    v.type = NUM;
    v.num = n;
    return v;
}

JValue JValue::make_bool(bool b) {
    JValue v;
    v.type = BOOL;
    v.b = b;
    return v;
}

JValue JValue::make_obj() {
    JValue v;
    v.type = OBJ;
    return v;
}

JValue JValue::make_arr() {
    JValue v;
    v.type = ARR;
    return v;
}

const JValue * JValue::find(const std::string & key) const {
    if (type != OBJ) {
        return nullptr;
    }
    for (const auto & kv : obj) {
        if (kv.first == key) {
            return &kv.second;
        }
    }
    return nullptr;
}

std::string JValue::get_str(const std::string & key, const std::string & def) const {
    const JValue * v = find(key);
    return (v && v->type == STR) ? v->str : def;
}

double JValue::get_num(const std::string & key, double def) const {
    const JValue * v = find(key);
    if (!v) {
        return def;
    }
    if (v->type == NUM) {
        return v->num;
    }
    if (v->type == STR) { // registries sometimes quote numbers
        char * end = nullptr;
        double d = strtod(v->str.c_str(), &end);
        if (end && *end == '\0' && end != v->str.c_str()) {
            return d;
        }
    }
    return def;
}

void JValue::set(const std::string & key, JValue v) {
    type = OBJ;
    for (auto & kv : obj) {
        if (kv.first == key) {
            kv.second = std::move(v);
            return;
        }
    }
    obj.emplace_back(key, std::move(v));
}

// ---- writer -------------------------------------------------------------

static void dump_string(const std::string & s, std::string & out) {
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
        }
    }
    out.push_back('"');
}

static void dump_value(const JValue & v, int indent, int depth, std::string & out) {
    auto newline = [&](int d) {
        if (indent >= 0) {
            out.push_back('\n');
            out.append((size_t) (indent * d), ' ');
        }
    };

    switch (v.type) {
        case JValue::NUL:
            out += "null";
            break;
        case JValue::BOOL:
            out += v.b ? "true" : "false";
            break;
        case JValue::NUM: {
            double d = v.num;
            if (std::isfinite(d) && d == std::floor(d) && std::fabs(d) < 9.0e15) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%lld", (long long) d);
                out += buf;
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.17g", d);
                out += buf;
            }
            break;
        }
        case JValue::STR:
            dump_string(v.str, out);
            break;
        case JValue::ARR: {
            out.push_back('[');
            for (size_t i = 0; i < v.arr.size(); i++) {
                if (i) {
                    out.push_back(',');
                }
                newline(depth + 1);
                dump_value(v.arr[i], indent, depth + 1, out);
            }
            if (!v.arr.empty()) {
                newline(depth);
            }
            out.push_back(']');
            break;
        }
        case JValue::OBJ: {
            out.push_back('{');
            for (size_t i = 0; i < v.obj.size(); i++) {
                if (i) {
                    out.push_back(',');
                }
                newline(depth + 1);
                dump_string(v.obj[i].first, out);
                out.push_back(':');
                if (indent >= 0) {
                    out.push_back(' ');
                }
                dump_value(v.obj[i].second, indent, depth + 1, out);
            }
            if (!v.obj.empty()) {
                newline(depth);
            }
            out.push_back('}');
            break;
        }
    }
}

std::string JValue::dump(int indent) const {
    std::string out;
    dump_value(*this, indent, 0, out);
    return out;
}

// ---- parser -------------------------------------------------------------

namespace {

struct Parser {
    const char * p;
    const char * end;
    std::string  err;
    int          depth = 0;

    bool error(const std::string & msg) {
        if (err.empty()) {
            err = msg;
        }
        return false;
    }

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
            p++;
        }
    }

    bool literal(const char * lit) {
        size_t n = strlen(lit);
        if ((size_t) (end - p) < n || strncmp(p, lit, n) != 0) {
            return error("invalid literal");
        }
        p += n;
        return true;
    }

    bool parse_hex4(unsigned & cp) {
        if (end - p < 4) {
            return error("truncated \\u escape");
        }
        cp = 0;
        for (int i = 0; i < 4; i++) {
            char c = *p++;
            cp <<= 4;
            if (c >= '0' && c <= '9') cp |= (unsigned) (c - '0');
            else if (c >= 'a' && c <= 'f') cp |= (unsigned) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') cp |= (unsigned) (c - 'A' + 10);
            else return error("bad \\u escape");
        }
        return true;
    }

    static void append_utf8(unsigned cp, std::string & s) {
        if (cp < 0x80) {
            s.push_back((char) cp);
        } else if (cp < 0x800) {
            s.push_back((char) (0xC0 | (cp >> 6)));
            s.push_back((char) (0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            s.push_back((char) (0xE0 | (cp >> 12)));
            s.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            s.push_back((char) (0x80 | (cp & 0x3F)));
        } else {
            s.push_back((char) (0xF0 | (cp >> 18)));
            s.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
            s.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            s.push_back((char) (0x80 | (cp & 0x3F)));
        }
    }

    bool parse_string(std::string & out) {
        if (p >= end || *p != '"') {
            return error("expected string");
        }
        p++;
        while (p < end) {
            unsigned char c = (unsigned char) *p;
            if (c == '"') {
                p++;
                return true;
            }
            if (c == '\\') {
                p++;
                if (p >= end) {
                    break;
                }
                char e = *p++;
                switch (e) {
                    case '"':  out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/');  break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        unsigned cp;
                        if (!parse_hex4(cp)) {
                            return false;
                        }
                        if (cp >= 0xD800 && cp <= 0xDBFF && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                            p += 2;
                            unsigned lo;
                            if (!parse_hex4(lo)) {
                                return false;
                            }
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                append_utf8(cp, out);
                                cp = lo;
                            }
                        }
                        append_utf8(cp, out);
                        break;
                    }
                    default:
                        return error("bad escape");
                }
            } else if (c < 0x20) {
                return error("control character in string");
            } else {
                out.push_back((char) c);
                p++;
            }
        }
        return error("unterminated string");
    }

    bool parse_value(JValue & out) {
        if (++depth > 256) {
            return error("nesting too deep");
        }
        skip_ws();
        if (p >= end) {
            return error("unexpected end of input");
        }
        bool ok;
        switch (*p) {
            case '{': {
                p++;
                out.type = JValue::OBJ;
                skip_ws();
                if (p < end && *p == '}') {
                    p++;
                    ok = true;
                    break;
                }
                while (true) {
                    skip_ws();
                    std::string key;
                    if (!parse_string(key)) {
                        return false;
                    }
                    skip_ws();
                    if (p >= end || *p != ':') {
                        return error("expected ':'");
                    }
                    p++;
                    JValue v;
                    if (!parse_value(v)) {
                        return false;
                    }
                    out.obj.emplace_back(std::move(key), std::move(v));
                    skip_ws();
                    if (p < end && *p == ',') {
                        p++;
                        continue;
                    }
                    if (p < end && *p == '}') {
                        p++;
                        break;
                    }
                    return error("expected ',' or '}'");
                }
                ok = true;
                break;
            }
            case '[': {
                p++;
                out.type = JValue::ARR;
                skip_ws();
                if (p < end && *p == ']') {
                    p++;
                    ok = true;
                    break;
                }
                while (true) {
                    JValue v;
                    if (!parse_value(v)) {
                        return false;
                    }
                    out.arr.push_back(std::move(v));
                    skip_ws();
                    if (p < end && *p == ',') {
                        p++;
                        continue;
                    }
                    if (p < end && *p == ']') {
                        p++;
                        break;
                    }
                    return error("expected ',' or ']'");
                }
                ok = true;
                break;
            }
            case '"':
                out.type = JValue::STR;
                ok = parse_string(out.str);
                break;
            case 't':
                out.type = JValue::BOOL;
                out.b = true;
                ok = literal("true");
                break;
            case 'f':
                out.type = JValue::BOOL;
                out.b = false;
                ok = literal("false");
                break;
            case 'n':
                out.type = JValue::NUL;
                ok = literal("null");
                break;
            default: {
                char * num_end = nullptr;
                double d = strtod(p, &num_end);
                if (num_end == p) {
                    return error("unexpected character");
                }
                out.type = JValue::NUM;
                out.num = d;
                p = num_end;
                ok = true;
            }
        }
        depth--;
        return ok;
    }
};

} // namespace

bool json_parse(const std::string & text, JValue & out, std::string & err) {
    Parser ps;
    ps.p = text.data();
    ps.end = text.data() + text.size();
    if (!ps.parse_value(out)) {
        err = ps.err.empty() ? "parse error" : ps.err;
        return false;
    }
    ps.skip_ws();
    if (ps.p != ps.end) {
        err = "trailing characters after JSON value";
        return false;
    }
    return true;
}

} // namespace alpacca
