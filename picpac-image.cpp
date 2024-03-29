#include "picpac-image.h"

namespace picpac {

    cv::Mat decode_raw (char const *buf, size_t sz) {
        if (sz < sizeof(int) * 4) return cv::Mat();
        int type = *reinterpret_cast<int const *>(buf); buf += sizeof(int); sz -= sizeof(int);
        int rows = *reinterpret_cast<int const *>(buf); buf += sizeof(int); sz -= sizeof(int);
        int cols = *reinterpret_cast<int const *>(buf); buf += sizeof(int); sz -= sizeof(int);
        int elemSize = *reinterpret_cast<int const *>(buf); buf += sizeof(int); sz -= sizeof(int);
        if (sz != size_t(rows * cols * elemSize)) return cv::Mat();
        cv::Mat m(rows, cols, type);
        if (int(m.elemSize()) != elemSize) return cv::Mat();
        size_t line = cols * elemSize;
        for (int i = 0; i < rows; ++i) {
            std::copy(buf, buf + line, m.ptr<char>(i));
            buf += line;
        }
        return m;
    }

    cv::Mat decode_buffer (string_view imbuf, int mode) {
        cv::Mat image = cv::imdecode(cv::Mat(1, imbuf.size(), CV_8U,
                        const_cast<char *>(imbuf.data())), mode);
        if (!image.data) {
            image = decode_raw(imbuf.data(), imbuf.size());
        }
        return image;
    }





    Annotation::Annotation (char const *begin, char const *end, cv::Size sz): size(sz) {
        //std::cerr << string(begin, end) << std::endl;
        if (begin) {
            json anno = json::parse(begin, end);
            int serial = 1;
            for (auto const &shape: anno.at("shapes")) {
                shapes.emplace_back(Shape::create(shape, size));
                shapes.back()->serial = serial++;
            }
        }
    }

    void ImageLoader::load (RecordReader rr, bool perturb, PerturbVector const &pv, Value *out,
           CacheValue *cache, std::mutex *mutex) const {
        Value cached;
        do {
            if (cache) { // access cache
                lock_guard lock(*mutex);
                cached.copy(*cache);
            }
            if (cached.facets.size() && cached.facets[0].image.data) break;
            if (cached.raw.size()) break;
            // cache miss, load the data
            Record r;
            rr(&r); // disk access
            cached.label = r.meta().label;
            cached.id = r.meta().id;
            for (int rf: config.raw) {
                cached.raw.push_back(r.field_string(rf));
            }
            for (int field: config.images) {
                //CHECK(r.size() >= (annotate ? 2 : 1));
                int decode_mode = cv::IMREAD_UNCHANGED;
                if (config.channels == 1) {
                    decode_mode = cv::IMREAD_GRAYSCALE;
                }
                else if (config.channels == 3) {
                    decode_mode = cv::IMREAD_COLOR;
                }
                cv::Mat image = decode_buffer(r.field(field), decode_mode);
                int channels = config.channels;
                if (channels < 0) channels = image.channels();
                int type = CV_MAKETYPE(config.dtype, channels);

                if (image.type() != type) {
                    cv::Mat tmp;
                    image.convertTo(tmp, type);
                    image = tmp;
                }

                cached.facets.emplace_back(image);
            }

            for (int field: config.annotate) {
                // load annotation
                int tt = r.fieldType(field);
                if (tt == 0) {
                    // guess field
                    string_view buf = r.field(field);
                    const char* p = buf.data();
                    int sz = buf.size();
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
                    string_view buf = r.field(field);
                    const char* p = buf.data();
                    int sz = buf.size();
                    cached.facets.emplace_back(p, p + sz, cached.facets.front().image.size());
                }
                else if (tt == FIELD_ANNOTATION_IMAGE) {
                    string_view buf = r.field(field);
                    cached.facets.emplace_back(decode_buffer(buf, -1));
                }
                else {
                    cached.facets.emplace_back(nullptr, nullptr, cached.facets.front().image.size());
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
        transforms.apply(&cached, perturb, &pv.buffer[0]);


        out->swap(cached);
    }
}

