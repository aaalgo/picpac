#include "picpac-image.h"

namespace picpac {

    Annotation::Annotation (char const *begin, char const *end, cv::Size sz): size(sz) {
        //std::cerr << string(begin, end) << std::endl;
        if (begin) {
            json anno = json::parse(begin, end);
            for (auto const &shape: anno.at("shapes")) {
                shapes.emplace_back(Shape::create(shape, size));
            }
        }
    }

    void ImageLoader::load (RecordReader rr, PerturbVector const &pv, Value *out,
           CacheValue *cache, std::mutex *mutex) const {
        Value cached;
        do {
            if (cache) { // access cache
                lock_guard lock(*mutex);
                cached.copy(*cache);
            }
            if (cached.facets.size() && cached.facets[0].image.data) break;
            // cache miss, load the data
            Record r;
            rr(&r); // disk access
            cached.label = r.meta().label;
            //CHECK(r.size() >= (annotate ? 2 : 1));
            int decode_mode = cv::IMREAD_UNCHANGED;
            if (config.channels == 1) {
                decode_mode = cv::IMREAD_GRAYSCALE;
            }
            else if (config.channels == 3) {
                decode_mode = cv::IMREAD_COLOR;
            }

            cached.facets.emplace_back(decode_buffer(r.field(0), decode_mode));

            if (config.annotate) {
                // load annotation
                int tt = r.fieldType(1);
                if (tt == 0) {
                    // guess field
                    const_buffer buf = r.field(1);
                    const unsigned char* p = boost::asio::buffer_cast<const unsigned char*>(buf);
                    int sz = boost::asio::buffer_size(buf);
                    if (sz > 0) {
                        for (int i = 0; i < sz; ++i) {
                            if (p[i] == ' ') continue;
                            if (p[i] == '\t') continue;
                            if (p[i] == '\n') continue;
                            if (p[i] == '{')  tt = FIELD_ANNOTATION_JSON;
                            else tt = FIELD_ANNOTATION_IMAGE;
                            break;
                        }
                    }
                }
                if (tt == FIELD_ANNOTATION_JSON) {
                    const_buffer buf = r.field(1);
                    const char* p = boost::asio::buffer_cast<const char*>(buf);
                    int sz = boost::asio::buffer_size(buf);
                    cached.facets.emplace_back(p, p + sz, cached.facets.back().image.size());
                }
                else if (tt == FIELD_ANNOTATION_IMAGE) {
                    const_buffer buf = r.field(1);
                    cached.facets.emplace_back(decode_buffer(buf, -1));
                }
                else {
                    cached.facets.emplace_back(nullptr, nullptr, cached.facets.back().image.size());
                }
                cached.facets.back().type = Facet::LABEL;
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

