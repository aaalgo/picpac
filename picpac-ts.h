#pragma once
#include <dlfcn.h>
#include <random>
#include <boost/core/noncopyable.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include "picpac.h"
#include "3rd/json/single_include/nlohmann/json.hpp"

#define PICPAC_CONFIG picpac::TimeSeriesStream::Config

#define PICPAC_CONFIG_UPDATE_ALL(C) \
    PICPAC_CONFIG_UPDATE(C,seed);\
    PICPAC_CONFIG_UPDATE(C,loop);\
    PICPAC_CONFIG_UPDATE(C,shuffle);\
    PICPAC_CONFIG_UPDATE(C,reshuffle);\
    PICPAC_CONFIG_UPDATE(C,stratify);\
    PICPAC_CONFIG_UPDATE(C,split);\
    PICPAC_CONFIG_UPDATE(C,split_fold);\
    PICPAC_CONFIG_UPDATE(C,split_negate);\
    PICPAC_CONFIG_UPDATE(C,mixin);\
    PICPAC_CONFIG_UPDATE(C,mixin_group_reset);\
    PICPAC_CONFIG_UPDATE(C,mixin_group_delta);\
    PICPAC_CONFIG_UPDATE(C,mixin_max);\
    PICPAC_CONFIG_UPDATE(C,cache);\
    PICPAC_CONFIG_UPDATE(C,preload);\
    PICPAC_CONFIG_UPDATE(C,threads);

namespace picpac {

	using json = nlohmann::json;

    struct Series {
        xt::xtensor<float, 1> time;    // (N,)
        xt::xtensor<float, 2> value;  // (N,C)

        void decode (const_buffer);

        float interp (Series const &s, float begin, int size, float step, bool);
    };

    struct Sample: private boost::noncopyable {
        uint32_t id;
        float label;
        float begin, end;
        string meta;
        vector<Series> series;  // multiple series

        Sample () {}

        int channels () const {
            int i = 0;
            for (auto const &s: series) {
                i += s.value.shape(1);
            }
            return i;
        }

        void swap (Sample &v) {
            std::swap(id, v.id);
            std::swap(label, v.label);
            std::swap(begin, v.begin);
            std::swap(end, v.end);
            std::swap(meta, v.meta);
            series.swap(v.series);
        }

        void copy (Sample const &v) {
            id = v.id;
            label = v.label;
            begin = v.begin;
            end = v.end;
            meta = v.meta;
            series = v.series;
        }

        Sample (Sample &&s) {
            swap(s);
        }
    };

    class Transform {
    public:
        static std::unique_ptr<Transform> create (json const &);
        virtual size_t pv_size () const {
            return 0;
        }
        virtual size_t pv_sample (random_engine &rng, void *pv) const {
            return 0;
        }
        virtual size_t apply (Sample *s, void const *pv) const {
            size_t sz = pv_size();
            /*
            for (auto &v: s->facets) {
                //size_t s = apply_one(&v, pv);
                //CHECK(s == sz);
            }
            */
            CHECK(0);   //TODO
            return sz;
        }

    };

    class Transforms: public Transform {
        int total_pv_size;
        vector<std::unique_ptr<Transform>> sub;
        friend class SomeOf;
    public:
        Transforms (json const &spec): total_pv_size(0) {
            for (auto it = spec.begin(); it != spec.end(); ++it) {
                sub.emplace_back(Transform::create(*it));
                total_pv_size += sub.back()->pv_size();
            }
        }
        virtual size_t pv_size () const {
            return total_pv_size;
        }
        virtual size_t pv_sample (random_engine &rng, void *pv) const {
            char *p = reinterpret_cast<char *>(pv);
            char *o = p;
            for (unsigned i = 0; i < sub.size(); ++i) {
                o += sub[i]->pv_sample(rng, o);
            }
            CHECK(total_pv_size == o-p);
            return total_pv_size;
        }
        virtual size_t apply (Sample *s, void const *pv) const {
            char const *p = reinterpret_cast<char const *>(pv);
            char const *o = p;
            for (unsigned i = 0; i < sub.size(); ++i) {
                o += sub[i]->apply(s, o);
            }
            CHECK(total_pv_size == o-p);
            return total_pv_size;
        }
    };


    class TimeSeriesLoader {
    public:
        struct Config {
            string transforms;
            Config ()
                : transforms("[]")
            {
            }
        };

        typedef Sample Value;

        typedef Sample CacheValue;

        struct PerturbVector {
            string buffer;
        };


        TimeSeriesLoader (Config const &c)
            : config(c), transforms(json::parse(c.transforms)), pv_size(transforms.pv_size())
        {
        }

        void sample (random_engine &e, PerturbVector *p) const {
            p->buffer.resize(pv_size);
            transforms.pv_sample(e, &p->buffer[0]);
        }

        void load (RecordReader, PerturbVector const &, Value *,
                CacheValue *c = nullptr, std::mutex *m = nullptr) const;

    protected:
        Config config;
        Transforms transforms;
        size_t pv_size;
    };

    typedef PrefetchStream<TimeSeriesLoader> TimeSeriesStream;
}

