#pragma once
#include "picpac.h"

namespace picpac {

    void sha1sum (char const *data, unsigned length, std::string *checksum);
    bool is_url (std::string const &url);

    class CachedDownloader {
        static string DEFAULT_AGENT;
        fs::path cache_dir;
        bool keep;
        int retry;
        int timeout;
        string agent;
        fs::path cache_path (const std::string &url) {
            string sum;
            sha1sum(&url[0], url.size(), &sum);
            return cache_dir / fs::path(sum);
        }
    public:
        CachedDownloader (fs::path const &cache = ".picpac_cache", bool keep_ = true)
            : cache_dir(cache), keep(keep_), retry(-1), timeout(-1), agent(DEFAULT_AGENT)
        {
            fs::create_directories(cache_dir);
        }
        ~CachedDownloader () {
            if (!keep) {
                remove_all(cache_dir);
            }
        }
        void set_timeout (int t) {
            timeout = t;
        }
        void set_agent (string const &a) {
            agent = a;
        }
        fs::path download (const std::string &url);
    };
}

