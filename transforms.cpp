#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include "picpac-ts.h"

namespace picpac {

    class Regularize: public Transform {
        int size;
        float step;
        float duration;
    public:
        Regularize (json const &spec)
            : size(spec.value<int>("size", -1)),
            step(spec.value<float>("step", -1)),
            duration(size * step) {
            CHECK(duration > 0);
            CHECK(step > 0);
        }
        virtual size_t pv_size () const {
            return 0;
        }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            return 0;
        }
        virtual size_t apply (Sample *s, void const *buf) const {
            CHECK(s->series.size() > 0);
            for (auto const &se: s->series) {
                if (se.time.shape(0) == 0) {
                    for (auto &ss: s->series) {
                        ss.time.resize({size});
                        ss.time.fill(0);   // TODO
                        ss.value.resize({size, ss.value.shape(1)});
                        ss.value.fill(0.0);
                    }
                    return 0;
                }
            }
            //spdlog::info("SZ: {}", s->series.size());
            float min_begin = s->series[0].time(0);
            float max_begin = min_begin;
            float min_end = s->series[0].time.periodic(-1);
            float max_end = min_end;
            for (int i = 1; i < s->series.size(); ++i) {
                auto &t = s->series[i].time;
                float begin = t(0);
                float end = t.periodic(-1);
                min_begin = std::min(min_begin, begin);
                max_begin = std::max(max_begin, begin);
                min_end = std::min(min_end, end);
                max_end = std::max(max_end, end);
            }
            float I = min_end - max_begin;  // intersection
            float U = max_end - min_begin;  // union
            float offset = -1;
            if (duration <= I) {
                offset = max_begin;
            }
            else if (duration <= U) {
                offset = max_begin - (duration - I);
                if (offset < min_begin) {
                    offset = min_begin;
                }
                CHECK(max_end - offset >= duration);
            }
            else {
                offset = min_begin - (duration - U)/2;
            }
            //spdlog::info("{} {} {} {} {}", min_begin, max_begin, min_end, max_end, offset);
            vector<Series> series(s->series.size());
            for (unsigned i = 0; i < series.size(); ++i) {
                series[i].interp(s->series[i], offset, size, step, false);
            }
            s->series.swap(series);
            return 0;
        }
    };

    class Mask: public Transform {
    public:
        Mask (json const &spec) {
        }
        virtual size_t apply (Sample *s, void const *buf) const {
            CHECK(s->series.size());
            s->series.emplace_back();
            Series &ss = s->series.back();
            ss.time = s->series[0].time;
            int size = ss.time.shape(0);
            ss.value.resize({size, 1});
            ss.value.fill(0.0);
            if (s->end < 0) return 0;
            int off = 0;
            while ((off < size) &&  (ss.time(off) < s->begin)) {
                ++off;
            }
            while ((off < size) &&  (ss.time(off) <= s->end)) {
                ss.value(off, 0) = 1.0;
                ++off;
            }
            return 0;
        }
    };

    class Rotation: public Transform {
        // random sample from SO(3)
        // and apply that globally to the seriest
        std::uniform_real_distribution<float> uniform;
        std::uniform_real_distribution<float> uniform_twopi;
        struct PerturbVector {
            float u1, u2, u3;
        };
    public:
        Rotation (json const &spec): uniform(0, 1), uniform_twopi(0, 2 * M_PI) {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->u1 = const_cast<Rotation *>(this)->uniform(rng);
            pv->u2 = const_cast<Rotation *>(this)->uniform_twopi(rng);
            pv->u3 = const_cast<Rotation *>(this)->uniform_twopi(rng);
            return sizeof(PerturbVector);
        }
        virtual size_t apply (Sample *s, void const *buf) const {
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            float alpha = std::sqrt(1-pv->u1);
            float beta = std::sqrt(pv->u1);
            float a = alpha * std::sin(pv->u2);
            float b = alpha * std::cos(pv->u2);
            float c = beta * std::sin(pv->u3);
            float d = beta * std::cos(pv->u3);
            float a2 = a * a;
            float b2 = b * b;
            float c2 = c * c;
            float d2 = d * d;
            float n2 = a2 + b2 + c2 + d2;
            CHECK(std::abs(n2 - 1.0) <= 0.001);
            float w = a, x = b, y = c, z = d;
            float w2 = a2, x2 = b2, y2 = c2, z2 = d2;
            xt::xtensor<float, 2> so = {
                {1.0 - 2 * y2 - 2 * z2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w},
                {2 * x * y + 2 * z * w, 1.0 - 2 * x2 - 2 * z2, 2 * y * z - 2 * x * w},
                {2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1.0 - 2 * x2 - 2 * y2}};
            for (auto &se: s->series) {
                if (se.value.shape(1) != 3) continue;
                se.value = xt::linalg::dot(se.value, so);
            }
            return sizeof(PerturbVector);
        }
    };

    class Noise: public Transform {
        // random sample from SO(3)
        // and apply that globally to the seriest
        float sigma;
        struct PerturbVector {
            int seed;
        };
    public:
        Noise (json const &spec) 
            : sigma(spec.value<float>("sigma", 0.01)) {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->seed = rng();
            return sizeof(PerturbVector);
        }
        virtual size_t apply (Sample *s, void const *buf) const {
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            std::default_random_engine rng(pv->seed);
            std::normal_distribution<float> gaussian(0, sigma);
            for (auto &se: s->series) {
                if (se.value.shape(1) != 3) continue;
                for (float &v: se.value) {
                    v *= 1.0 + gaussian(rng);
                }
            }
            return sizeof(PerturbVector);
        }
    };

    class Shift: public Transform {
        // random sample from SO(3)
        // and apply that globally to the seriest
        float sigma;
        struct PerturbVector {
            int seed;
        };
    public:
        Shift (json const &spec) 
            : sigma(spec.value<float>("sigma", 0.01)) {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->seed = rng();
            return sizeof(PerturbVector);
        }
        virtual size_t apply (Sample *s, void const *buf) const {
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            std::default_random_engine rng(pv->seed);
            std::normal_distribution<float> gaussian(0, sigma);
            for (auto &se: s->series) {
                if (se.value.shape(1) != 3) continue;
                xt::xtensor<float, 1> t = se.time;
                for (int i = 1; i < t.shape(0) - 1; ++i) {
                    float left = t(i-1);
                    float mid = t(i);
                    float right = t(i+1);
                    left = mid - (mid - left) / 3;
                    right = mid + (right - mid) / 3;
                    mid = mid * (1 + gaussian(rng));
                    se.time(i) = std::min(std::max(left, mid), right);
                }
            }
            return sizeof(PerturbVector);
        }
    };

    class Clip: public Transform {
        std::uniform_real_distribution<float> uniform;
        float ratio;
        bool is_train;
        struct PerturbVector {
            float left;
        };
    public:
        Clip (json const &spec) 
            : uniform(0, 1.0),
              ratio(spec.value<float>("ratio", 0.05)),
              is_train(spec.value<bool>("train", true))
        {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            if (is_train) {
                pv->left = const_cast<Clip *>(this)->uniform(rng);
            }
            else {
                pv->left = 0.5;
            }
            return sizeof(PerturbVector);
        }
        virtual size_t apply (Sample *s, void const *buf) const {
            if (ratio == 0) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            int size0 = s->series[0].time.shape(0);
            int clip = int(std::roundf(ratio * size0));
            int left = int(std::roundf(clip * pv->left));
            CHECK(clip < size0);
            if (left <= 0) left = 1;
            else if (left >= clip) left = clip - 1;

            int size = size0 - clip;
            for (auto &se: s->series) {
                CHECK(size0 == se.time.shape(0));
                se.time = xt::view(se.time, xt::range(left, left + size));
                se.value = xt::view(se.value, xt::range(left, left + size), xt::all());
            }
            return sizeof(PerturbVector);
        }
    };

    class Sometimes: public Transform {
        float chance;
        Transforms transforms;
        std::uniform_real_distribution<float> uniform;
        struct PerturbVector {
            bool apply;
        };
    public:
        Sometimes (json const &spec)
            : chance(spec.value<float>("chance", 0.5)),
            transforms(spec.at("transforms")),
            uniform(0.0, 1.0) {
        }
        virtual size_t pv_size () const {
            return sizeof(PerturbVector) + transforms.pv_size(); 
        }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            float p = const_cast<Sometimes *>(this)->uniform(rng);
            pv->apply = p <= chance;
            if (pv->apply) {
                transforms.pv_sample(rng, static_cast<char*>(buf) + sizeof(PerturbVector));
            }
            return pv_size();
        }
        virtual size_t apply (Sample *s, void const *buf) const {
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            if (pv->apply) {
                transforms.apply(s, static_cast<char const*>(buf) + sizeof(PerturbVector));
            }
            return pv_size();
        }
    };

    class SomeOf: public Transform {
        static const unsigned MAX = 32;
        struct PerturbVector {
            std::array<bool, MAX> apply;
        };
        unsigned count;
        Transforms transforms;
    public:
        SomeOf (json const &spec)
            : count(spec.value<unsigned>("count", 1)),
            transforms(spec.at("transforms")) {
            CHECK(transforms.sub.size() < MAX);
            CHECK(count <= transforms.sub.size());
        }
        virtual size_t pv_size () const {
            return sizeof(PerturbVector) + transforms.pv_size(); 
        }

        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);

            std::vector<unsigned> index(transforms.sub.size());
            for (unsigned i = 0; i < index.size(); ++i) {
                index[i] = i;
            }
            std::shuffle(index.begin(), index.end(), rng);
            std::fill(pv->apply.begin(), pv->apply.end(), false);
            for (unsigned i = 0; i < count; ++i) {
                pv->apply[index[i]] = true;
            }
            char *p = static_cast<char*>(buf) + sizeof(PerturbVector);
            for (unsigned i = 0; index.size(); ++i) {
                if (pv->apply[i]) {
                    p += transforms.sub[i]->pv_sample(rng, p);
                }
            }
            return pv_size();
        }

        virtual size_t apply (Sample *s, void const *buf) const {
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            char const *p = static_cast<char const *>(buf) + sizeof(PerturbVector);
            for (unsigned i = 0; i < transforms.sub.size(); ++i) {
                if (pv->apply[i]) {
                    p += transforms.sub[i]->apply(s, p);
                }
            }
            return pv_size();
        }
    };


    typedef Transform *(*transform_factory_t) (json const &spec);
    vector<transform_factory_t> transform_factories;

    std::unique_ptr<Transform> Transform::create (json const &spec) {
        string type = spec.at("type").get<string>();
        for (auto fun: transform_factories) {
            Transform *p = fun(spec);
            if (p) return std::unique_ptr<Transform>(p);
        }
        if (type == "sometimes") {
            return std::unique_ptr<Transform>(new Sometimes(spec));
        }
        if (type == "some_of") {
            return std::unique_ptr<Transform>(new SomeOf(spec));
        }
        if (type == "regularize") {
            return std::unique_ptr<Transform>(new Regularize(spec));
        }
        if (type == "mask") {
            return std::unique_ptr<Transform>(new Mask(spec));
        }
        if (type == "rotation") {
            return std::unique_ptr<Transform>(new Rotation(spec));
        }
        if (type == "noise") {
            return std::unique_ptr<Transform>(new Noise(spec));
        }
        if (type == "shift") {
            return std::unique_ptr<Transform>(new Shift(spec));
        }
        if (type == "clip") {
            return std::unique_ptr<Transform>(new Clip(spec));
        }
        CHECK(0);
        return nullptr;
    }

    void load_transform_library (string const &path) {
        void *h = dlopen(path.c_str(), RTLD_LAZY);
        CHECK(h);   // intentionally left unclosed
        transform_factory_t fun;
        void *sym = dlsym(h, "transform_factory");
        CHECK(sym); // << "Failed to load " << path << ": " << dlerror();
        *(void**)(&fun) = sym;
        transform_factories.push_back(fun);
    }
}
