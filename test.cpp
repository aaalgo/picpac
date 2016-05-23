#define CATCH_CONFIG_MAIN
#include <map>
#include <set>
#include <functional>
#include <algorithm>
#include <iterator>
#include <catch.hpp>
#include "picpac.h"
#include "picpac-cv.h"

using namespace std;
using namespace picpac;

struct TestConfig: PICPAC_CONFIG {
    unsigned class_count;
    unsigned write_count;
    unsigned read_count;
    TestConfig ()
        : class_count(5),
        write_count(853),
        read_count(3461*11-1) {
    }
};

class TestDB {
    fs::path path;
public:
    TestDB (): path(fs::unique_path()) {
        TestConfig config;
        WARN("Creating test db: " << path);
        FileWriter out(path);
        for (unsigned i = 0; i < config.write_count; ++i) {
            Record r(0, lexical_cast<string>(i), string());
            r.meta().id = i;
            r.meta().label = int(sqrt(i)) % config.class_count;
            out.append(r);
        }
    }

    void loop (TestConfig const &config, function<void(Record const &)> f) const {
        Stream stream(path, config);
        map<uint32_t, int> cnt;
        for (unsigned i = 0; i < config.read_count; ++i) {
            Record r;
            stream.read_next(&r);
            f(r);
        }
    }

    void count_id (TestConfig const &config, map<uint32_t, int> *cnt) const {
        cnt->clear();
        loop(config, [cnt](Record const &r) {
                    ++(*cnt)[r.meta().id];
                });
    }

    void count_label (TestConfig const &config, map<uint32_t, int> *cnt) const {
        cnt->clear();
        loop(config, [cnt](Record const &r) {
                    ++(*cnt)[r.meta().label];
                });
    }

    void idset (TestConfig const &config, set<uint32_t> *v) const {
        v->clear();
        loop(config, [v](Record const &r) {
                    v->insert(r.meta().id);
                });
    }

    void labelset (TestConfig const &config, set<uint32_t> *v) const {
        v->clear();
        loop(config, [v](Record const &r) {
                    v->insert(r.meta().label);
                });
    }

    ~TestDB () {
        fs::remove(path);
    }
};

int inbalance (map<uint32_t, int> const &cnt) {
    int min = numeric_limits<int>::max();
    int max = 0;
    for (auto p: cnt) {
        if (p.second < min) min = p.second;
        if (p.second > max) max = p.second;
    }
    return max - min;
}

TEST_CASE("split streaming", "") {
    TestDB db;
    TestConfig config;
    config.loop = true;
    config.shuffle = true;
    config.split = 7;
    config.split_fold = 3;
    config.split_negate = false;

    SECTION("Stratify") {
        config.stratify = true;
        std::map<uint32_t, int> cnt;
        db.count_label(config, &cnt);
        REQUIRE(inbalance(cnt) <= 1);
    }

    SECTION("Non Stratify") {
        config.stratify = false;
        std::map<uint32_t, int> cnt;
        db.count_label(config, &cnt);
        REQUIRE(inbalance(cnt) > 1);
        db.count_id(config, &cnt);   // ID should be in balance
        REQUIRE(inbalance(cnt) <= 1);
    }

    SECTION("Cross validation -- ID") {
        set<uint32_t> train, val;
        vector<uint32_t> common;
        db.idset(config, &train);
        config.split_negate = true;
        db.idset(config, &val);
        set_intersection(train.begin(), train.end(),
                         val.begin(), val.end(),
                         back_inserter(common));
        REQUIRE(train.size() + val.size() == config.write_count);
        REQUIRE(common.empty());
        //  n/split ~ n/split + 1
        //  vs n * (split-1)/split
        REQUIRE(train.size() >= val.size());
    }

    SECTION("Cross validation -- Label") {
        set<uint32_t> train, val;
        vector<uint32_t> common;
        db.labelset(config, &train);
        config.split_negate = true;
        db.labelset(config, &val);
        REQUIRE(train.size() == config.class_count);
        REQUIRE(val.size() == config.class_count);
    }
}

