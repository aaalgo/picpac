#include <sstream>
#include <string>
#include <exception>
#include <unordered_map>
#include <boost/lexical_cast.hpp>

namespace rfc3986 {
    using std::string;
    using std::unordered_map;
    using boost::lexical_cast;

// code from served
// https://github.com/datasift/served/blob/master/src/served/uri.cpp
// by MIT license
static const char hex_table[] = "0123456789ABCDEF";

static const char dec_to_hex[256] = {
    /*       0  1  2  3   4  5  6  7   8  9  A  B   C  D  E  F */
    /* 0 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 1 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 2 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 3 */  0, 1, 2, 3,  4, 5, 6, 7,  8, 9,-1,-1, -1,-1,-1,-1,

    /* 4 */ -1,10,11,12, 13,14,15,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 5 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 6 */ -1,10,11,12, 13,14,15,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 7 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,

    /* 8 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 9 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* A */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* B */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,

    /* C */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* D */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* E */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* F */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1
};

std::string
query_escape(const std::string& s) {
    const unsigned char* src_ptr = (const unsigned char*)s.c_str();
    const size_t         src_len = s.length();

    unsigned char        tmp[src_len * 3];
    unsigned char*       end = tmp;

    const unsigned char * const eol = src_ptr + src_len;

    for (; src_ptr < eol; ++src_ptr) {
        unsigned char c = *src_ptr;
        if ((c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '-' || c == '.' || c == '_' || c == '~')
        {
            *end++ = *src_ptr;
        } else {
            *end++ = '%';
            *end++ = hex_table[*src_ptr >> 4];
            *end++ = hex_table[*src_ptr & 0x0F];
        }
    }
    return std::string((char*)tmp, (char*)end);
}

std::string
query_unescape(const std::string& s) {
    const unsigned char* src_ptr = (const unsigned char*)s.c_str();
    const size_t         src_len = s.length();

    const unsigned char* const eol = src_ptr + src_len;
    const unsigned char* const last_decodable = eol - 2;

    char  tmp[src_len];
    char* end = tmp;

    while (src_ptr < last_decodable) {
        if (*src_ptr == '%') {
            char dec1, dec2;
            if (-1 != (dec1 = dec_to_hex[*(src_ptr + 1)])
                && -1 != (dec2 = dec_to_hex[*(src_ptr + 2)]))
            {
                *end++ = (dec1 << 4) + dec2;
                src_ptr += 3;
                continue;
            }
        }
        *end++ = *src_ptr++;
    }

    while (src_ptr < eol) {
        *end++ = *src_ptr++;
    }

    return std::string((char*)tmp, (char*)end);
}

    class Exception: public std::exception {
    };

    class Form: public unordered_map<string, string> {
    public:
        Form () {}
        Form (string const &query) {
            char const *b = query.empty() ? nullptr : &query[0];
            char const *e = b + query.size();
            while (b < e) {
                char const *sep = b;
                while (sep < e && *sep != '&') ++sep;
                char const *eq = b;
                while (eq < sep && *eq != '=') ++eq;
                // b  <= eq <= sep <= e
                //       =      &
                if (!(b < eq)) throw Exception();
                if (!(eq + 1 < sep)) throw Exception();
                string key = query_unescape(string(b, eq));
                string value = query_unescape(string(eq + 1, sep));
                insert(std::make_pair(key, value));
                b = sep + 1;
            }
        }
        template <typename T>
        T get (string const &key, T def) {
            auto it = find(key);
            if (it == end()) return def;
            return lexical_cast<T>(it->second);
        }

        string encode (bool amps = false) const {
            std::ostringstream ss;
            bool first = !amps;
            for (auto p: *this) {
                if (first) first = false;
                else ss << '&';
                ss << query_escape(p.first) << '=' << query_escape(p.second);
            }
            return ss.str();
        }
    };
}
