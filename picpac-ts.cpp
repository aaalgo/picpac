#include "picpac-ts.h"

namespace picpac {

    void Series::decode (const_buffer buffer) {
        // layout:
        // uint32_t size, channels
        // size x float
        // size x channels
        char const *buf = boost::asio::buffer_cast<char const *>(buffer);
        char const *off = buf;
        auto sz = boost::asio::buffer_size(buffer);
        uint32_t size = *reinterpret_cast<uint32_t const *>(off); off += sizeof(uint32_t);
        uint32_t channels = *reinterpret_cast<uint32_t const *>(off); off += sizeof(uint32_t);
        time.resize({size});
        value.resize({size, channels});

        for (int i = 0; i < time.shape(0); ++i) {
            time(i) = *reinterpret_cast<float const *>(off); off += sizeof(float);
            for (int j = 0; j < value.shape(1); ++j) {
                value(i, j) = *reinterpret_cast<float const *>(off); off += sizeof(float);
            }
        }
        /*
        memcpy(time.data(), off, size * sizeof(float));
        off += size * sizeof(float);
        memcpy(value.data(), off, size * channels * sizeof(float));
        off += size * channels * sizeof(float);
        */
        CHECK(off - buf == sz);
        int next = 0;
        for (int i = 0; i < size; ++i) {
            if (!std::isnormal(time(i))) continue;
            bool good = true;
            for (int j = 0; j < channels; ++j) {
                if (!std::isnormal(value(i, j))) {
                    good = false;
                    break;
                }
            }
            if (!good) continue;
            if (i > next) {
                time(next) = time(i);
                for (int j = 0; j < channels; ++j) {
                    value(next, j) = value(i, j);
                }
            }
            ++next;
        }
        if (next < size) {
            xt::xtensor<float, 1> time1;
            time1.resize({next});
            xt::xtensor<float, 2> value1;
            value1.resize({next, channels});
            std::copy(time.data(), time.data() + next, time1.data());
            std::copy(value.data(), value.data() + next * channels, value1.data());
            std::swap(time1, time);
            std::swap(value1, value);
        }
    }

    float Series::interp (Series const &s, float begin, int size, float step, bool extrapolate) {
        int c = s.value.shape(1);
        CHECK(s.time.shape(0) > 2);
        CHECK(s.time.shape(0) == s.value.shape(0));
        time.resize({size});
        value.resize({size, c});
        value.fill(0.0);

        float max_gap = 0;

        int off = 0;
        while (off < size) {
            float t = begin + step * off;
            if (t >= s.time(0)) break;
            time(off) = t;
            if (extrapolate) {
            // value, extrapoloate
            //  t   s.time(0)   s.time(1)
            //  v   s.value(0)  s.value(1)
            int left = 0, right = 1;
            float lb = s.time(left), ub = s.time(right);
            for (int i = 0; i < c; ++i) {
                value(off, i) = (s.value(left, i) * (ub -t) + s.value(right, i) * (t-lb)) / (ub - lb);
            }
            max_gap = std::max(max_gap, lb - t);
            }
            ++off;
        }
        int off1 = off;

        int left = 0, n0 = s.time.shape(0);
        CHECK(n0 == s.value.shape(0));
        while (off < size) {
            float t = begin + step * off;
            // move forward until left <= t < right
            int right = left + 1;
            while ((right < n0) && (s.time(right) <= t)) {
                left = right;
                right = left + 1;
            }
            if (right >= n0) break;
            // interpolate
            while  ((right < n0) && (s.time(right) <= s.time(left))) ++right;
            if (right >= n0) break;
            float lb = s.time(left), ub = s.time(right);
            CHECK((lb <= t) && (t < ub));
            time(off) = t;
            for (int i = 0; i < c; ++i) {
                value(off, i) = (s.value(left, i) * (ub -t) + s.value(right, i) * (t-lb)) / (ub - lb);
            }
            max_gap = std::max(max_gap, std::min(ub -t, t-lb));
            ++off;
        }
        int off2 = off;

        while (off < size) {
            float t = begin + step * off;
            time(off) = t;
            if (extrapolate) {
            int left = n0-2, right = n0-1;
            float lb = s.time(left), ub = s.time(right);
            for (int i = 0; i < c; ++i) {
                value(off, i) = (s.value(left, i) * (ub -t) + s.value(right, i) * (t-lb)) / (ub - lb);
            }
            max_gap = std::max(max_gap, t - ub);
            }
            ++off;
        }
        /*
        std::cout << off1 << " " << off2 << std::endl;
        std::cout << begin << " " << s.time(0) << std::endl;
        std::cout << value << std::endl;
        */
        return max_gap;
    }

    void TimeSeriesLoader::load (RecordReader rr, PerturbVector const &pv, Value *out,
           CacheValue *cache, std::mutex *mutex) const {
        Value cached;
        do {
            if (cache) { // access cache
                lock_guard lock(*mutex);
                cached.copy(*cache);
            }
            // CHECK cache load success
            if (cached.series.size()) break;
            // cache miss, load the data
            Record r;
            rr(&r); // disk access
            cached.label = r.meta().label;
            cached.id = r.meta().id;

            CHECK(r.size() >= 1);

            // load json
            {
                const_buffer buf = r.field(0);
                const char* b = boost::asio::buffer_cast<const char*>(buf);
                const char* e = b + boost::asio::buffer_size(buf);
                json anno = json::parse(b, e);

                cached.begin = cached.end = -1;
                cached.meta = string(b, e);

                if (anno.contains("range")) {
                    json range = anno.at("range");
                    cached.begin = range[0].get<float>();
                    cached.end = range[1].get<float>();
                }
            }

            cached.series.resize(r.size() - 1);

            for (unsigned i = 0; i < cached.series.size(); ++i) {
                cached.series[i].decode(r.field(i+1));
            }

            if (cache) {
                // write to cache
                lock_guard lock(*mutex);
                cache->copy(cached);
            }
        } while (false);
        // apply transformation
        transforms.apply(&cached, &pv.buffer[0]);
        out->swap(cached);
    }
}

