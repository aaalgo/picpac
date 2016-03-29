#include <string>
#include <exception>
#include <unordered_map>
#include <served/served.hpp>

namespace rfc3986 {
    using std::string;
    using std::unordered_map;
    using served::query_unescape;

    class Exception: public std::exception {
    };

    class Form: public unordered_map<string, string> {
    public:
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
    };
}
