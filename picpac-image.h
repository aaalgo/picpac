#pragma once
#include <random>
#include <boost/core/noncopyable.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "picpac.h"
#include "3rdparty/json.hpp"

#define PICPAC_CONFIG picpac::ImageStream::Config

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
    PICPAC_CONFIG_UPDATE(C,threads);\
    PICPAC_CONFIG_UPDATE(C,channels);\
    PICPAC_CONFIG_UPDATE(C,annotate);

namespace picpac {

    static inline int dtype_np2cv (string const &dt) {
        if (dt == "uint8") {
            return CV_8U;
        }
        else if (dt == "float32") {
            return CV_32F;
        }
        else if (dt == "int32") {
            return CV_32S;
        }
        else CHECK(false) << "dtype not supported.";
        return 0;
    }


	using json = nlohmann::json;

    struct RenderOptions {
        int thickness;
        int line_type;
        int shift;
        int point_radius;
        bool show_numbers;
        bool use_palette;
        bool use_tag;
        bool use_serial;

        RenderOptions ()
            : thickness(CV_FILLED),
            line_type(8),
            shift(0),
            point_radius(5),
            show_numbers(false),
            use_palette(false),
            use_tag(false),
            use_serial(false)
        {
        }
    };

    // One annotation shape, box or polygon.
    // Each shape must be controled by a series of control points
    // geometric transformation of shape can be achieved by applying transformations
    // to control points (with a callback function)
    class Shape {
    protected:
        vector<cv::Point2f> controls;
    public:
        char const *type;   // text name
        cv::Scalar color;
        float tag;
        int serial;

        cv::Scalar render_color (RenderOptions const &) const;

        Shape (char const *t): type(t), color(1.0, 1.0, 1.0, 1.0), tag(0) {}
        virtual ~Shape () {}
        virtual std::unique_ptr<Shape> clone () const = 0;
        virtual void transform (std::function<void(vector<cv::Point2f> *)> f) {
            // some shape might need pre-post processing
            f(&controls);
        }
        virtual void render (cv::Mat *, RenderOptions const &) const = 0;
        static std::unique_ptr<Shape> create (json const &, cv::Size);

        vector<cv::Point2f> const &__controls () const {
            return controls;
        };
    };

    struct Annotation: private boost::noncopyable {
        cv::Size size;
        vector<std::unique_ptr<Shape>> shapes;

        Annotation (): size(0,0) {}
        Annotation (char const *begin, char const *end, cv::Size);
        bool empty () const { return size.width == 0;}
        void clear () { size = cv::Size(0, 0); shapes.clear(); }

        void transform (std::function<void(vector<cv::Point2f> *)> f) {
            for (auto &s: shapes) {
                s->transform(f);
            }
        }

        void render (cv::Mat *m, RenderOptions const &opt) const {
            for (auto &s: shapes) {
                s->render(m, opt);
            }
        }

        void copy (Annotation const &anno) {
            size = anno.size;
            shapes.clear();
            for (auto &p: anno.shapes) {
                shapes.emplace_back(p->clone());
            }
        }

        void swap (Annotation &anno) {
            std::swap(size, anno.size);
            shapes.swap(anno.shapes);
        }
    };

    // image with annotation
    //
    struct Facet {
        enum {
            IMAGE = 1,
            LABEL = 2,
            FEATURE = 3,
            NONE = 4
        };
        int type;
        cv::Mat image;
        Annotation annotation;

        Facet (): type(IMAGE) {}

        Facet (char const *begin, char const *end, cv::Size sz): type(LABEL),
            annotation(begin, end, sz) {
        }

        Facet (cv::Mat v, int type_ = IMAGE): type(type_), image(v) {
        }

        Facet (Facet &&ai) {
            std::swap(type, ai.type);
            cv::swap(image, ai.image);
            annotation.swap(ai.annotation);
        }

        // ensure that all annotation has been cleared so data
        // can be converted to python array
        void check_pythonize () {
            if (type == LABEL) {
                if (!annotation.empty() || annotation.shapes.size()) {
                    CHECK(0) << "Your data contains a facet that cannot be converted to numpy array."
                        "This is because the label facet contains annotations that has not been"
                        " rasterized.  This can be avoided by adding a 'drop' transformation to discard the annotation or "
                        " a 'rasterize' transformation.";

                }
            }
        }

        void operator = (Facet &&ai) {
            std::swap(type, ai.type);
            cv::swap(image, ai.image);
            annotation.swap(ai.annotation);
        }

        cv::Size check_size () const {
            cv::Size image_sz(0,0);
            if (image.data) {
                image_sz = image.size();
            }
            cv::Size anno_sz = annotation.size;
            if (image_sz.width > 0 && anno_sz.width == 0) return image_sz;
            if (image_sz.width == 0 && anno_sz.width > 0) return anno_sz;
            if (image_sz.width > 0 && anno_sz.width > 0) {
                CHECK(image_sz == anno_sz);
                return image_sz;
            }
            if (image_sz.width == 0 && anno_sz.width ==0) {
                return cv::Size(0, 0);
            }
            CHECK(0);
            return image_sz;
        }
    private:
        Facet (Facet &) = delete;
        void operator = (Facet &) = delete;
    };

    struct Sample: private boost::noncopyable {
        uint32_t id;
        float label;
        vector<string> raw;
        vector<Facet> facets;

        Sample () {}

        void swap (Sample &v) {
            std::swap(id, v.id);
            std::swap(label, v.label);
            raw.swap(v.raw);
            facets.swap(v.facets);
        }

        void copy (Sample const &v) {
            id = v.id;
            label = v.label;
            raw.clear();
            for (auto const &s: v.raw) {
                raw.push_back(s);
            }
            facets.clear();
            facets.resize(v.facets.size());
            for (unsigned i = 0; i < facets.size(); ++i) {
                auto const &vi = v.facets[i];
                facets[i].type = vi.type;
                if (vi.image.data) {
                    facets[i].image = vi.image.clone();
                } 
                facets[i].annotation.copy(vi.annotation);
            }
        }

        Sample (Sample &&s) {
            swap(s);
        }
#if 0
    private:
        Sample (Sample &) = delete;
        void operator = (Sample &) = delete;
#endif
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
            for (auto &v: s->facets) {
                size_t s = apply_one(&v, pv);
                CHECK(s == sz);
            }
            return sz;
        }

        virtual size_t apply_one (Facet *, void const *) const {
            return 0;
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


    class ImageLoader {
    public:
        struct Config {
            int channels;       // -1: unchanged
            int dtype;
            vector<int> images;
            vector<int> annotate;
            vector<int> raw;
            //bool annotate;
            string transforms;
            Config ()
                : channels(-1), // unchanged
                images{0},
                dtype(CV_32F),
                annotate(false),
                transforms("[]")

            {
                CHECK(channels == -1 || channels == 1 || channels == 3);
            }
        };

        typedef Sample Value;

        typedef Sample CacheValue;

        struct PerturbVector {
            string buffer;
        };


        ImageLoader (Config const &c)
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

    typedef PrefetchStream<ImageLoader> ImageStream;

#if 0

    namespace impl {
        template <typename Tfrom = uint8_t, typename Tto = float>
        Tto *split_helper (cv::Mat image, Tto *buffer, cv::Scalar mean, bool bgr2rgb) {
            Tto *ptr_b = buffer;
            Tto *ptr_g = buffer;
            Tto *ptr_r = buffer;
            if (bgr2rgb) {
                CHECK(image.channels() == 3);
                ptr_g += image.total();
                ptr_b += 2 * image.total();
            }
            else if (image.channels() == 2) {
                ptr_g += image.total();     // g, r
            }
            else if (image.channels() == 3) {
                ptr_g += image.total();     // b, g, r
                ptr_r += 2 * image.total();
            }
            unsigned off = 0;
            for (int i = 0; i < image.rows; ++i) {
                Tfrom const *line = image.ptr<Tfrom const>(i);
                for (int j = 0; j < image.cols; ++j) {
                    ptr_b[off] = (*line++) - mean[0];
                    if (image.channels() > 1) {
                        ptr_g[off] = (*line++) - mean[1];
                    }
                    if (image.channels() > 2) {
                        ptr_r[off] = (*line++) - mean[2];
                    }
                    ++off;
                }
            }
            CHECK(off == image.total());
            return buffer + image.channels() * image.total();
        }

        template <typename Tto = float>
        Tto *split_copy (cv::Mat image, Tto *buffer, cv::Scalar mean, bool bgr2rgb) {
            int depth = image.depth();
            int ch = image.channels();
            CHECK((ch >= 1) && (ch <= 3));
            switch (depth) {
                case CV_8U: return split_helper<uint8_t, Tto>(image, buffer, mean, bgr2rgb);
                case CV_8S: return split_helper<int8_t, Tto>(image, buffer, mean, bgr2rgb);
                case CV_16U: return split_helper<uint16_t, Tto>(image, buffer, mean, bgr2rgb);
                case CV_16S: return split_helper<int16_t, Tto>(image, buffer, mean, bgr2rgb);
                case CV_32S: return split_helper<int32_t, Tto>(image, buffer, mean, bgr2rgb);
                case CV_32F: return split_helper<float, Tto>(image, buffer, mean, bgr2rgb);
                case CV_64F: return split_helper<double, Tto>(image, buffer, mean, bgr2rgb);
            }
            CHECK(0) << "Mat type not supported.";
            return nullptr;
        }

        template <typename Tfrom = uint8_t, typename Tto = float>
        Tto *copy_helper (cv::Mat image, Tto *buffer, cv::Scalar mean, bool bgr2rgb) {
            CHECK(!bgr2rgb);
            unsigned off = 0;
            for (int i = 0; i < image.rows; ++i) {
                Tfrom const *line = image.ptr<Tfrom const>(i);
                for (int j = 0; j < image.cols; ++j) {
                    buffer[off++] = (*line++) - mean[0];
                    if (image.channels() > 1) {
                        buffer[off++] = (*line++) - mean[1];
                    }
                    if (image.channels() > 2) {
                        buffer[off++] = (*line++) - mean[2];
                    }
                }
            }
            CHECK(off == image.total() * image.channels());
            return buffer + image.channels() * image.total();
        }

        template <typename Tto = float>
        Tto *copy (cv::Mat image, Tto *buffer, cv::Scalar mean, bool bgr2rgb) {
            CHECK(!bgr2rgb) << "Not supported";
            int depth = image.depth();
            int ch = image.channels();
            CHECK((ch >= 1) && (ch <= 3));
            switch (depth) {
                case CV_8U: return copy_helper<uint8_t, Tto>(image, buffer, mean, bgr2rgb);
                case CV_8S: return copy_helper<int8_t, Tto>(image, buffer, mean, bgr2rgb);
                case CV_16U: return copy_helper<uint16_t, Tto>(image, buffer, mean, bgr2rgb);
                case CV_16S: return copy_helper<int16_t, Tto>(image, buffer, mean, bgr2rgb);
                case CV_32S: return copy_helper<int32_t, Tto>(image, buffer, mean, bgr2rgb);
                case CV_32F: return copy_helper<float, Tto>(image, buffer, mean, bgr2rgb);
                case CV_64F: return copy_helper<double, Tto>(image, buffer, mean, bgr2rgb);
            }
            CHECK(0) << "Mat type not supported.";
            return nullptr;
        }

        
        template <typename Tfrom, typename Tto = float>
        Tto *onehot_helper (cv::Mat image, Tto *buffer, unsigned onehot, bool cf) {
            size_t ch_size = image.total();
            size_t total_size = ch_size * onehot;
            Tto *buffer_end = buffer + total_size;
            std::fill(buffer, buffer_end, 0);
            int off = 0;
            if (cf) { // channel comes first
                for (int i = 0; i < image.rows; ++i) {
                    Tfrom const *row = image.ptr<Tfrom const>(i);
                    for (int j = 0; j < image.cols; ++j) {
                        Tfrom v = row[j];
                        unsigned c(v);
                        CHECK(c == v);
                        CHECK(c <= MAX_CATEGORIES);
                        if (c < ch_size) {
                            Tto *plane = buffer + c * ch_size;
                            plane[off] = 1;
                        }
                        ++off;
                    }
                }
            }
            else { // channel comes last
                Tto *o = buffer;
                for (int i = 0; i < image.rows; ++i) {
                    Tfrom const *row = image.ptr<Tfrom const>(i);
                    for (int j = 0; j < image.cols; ++j) {
                        Tfrom v = row[j];
                        unsigned c(v);
                        CHECK(c == v);
                        CHECK(c <= MAX_CATEGORIES);
                        if (c < ch_size) {
                            o[c] = 1;
                        }
                        o += ch_size;
                    }
                }
            }
            return buffer_end;
        }

        template <typename Tto = float>
        Tto *onehot_encode (cv::Mat image, Tto *buffer, unsigned onehot, bool cf) {
            int depth = image.depth();
            int ch = image.channels();
            CHECK(ch == 1);
            switch (depth) {
                case CV_8U: return onehot_helper<uint8_t, Tto>(image, buffer, onehot, cf);
                case CV_8S: return onehot_helper<int8_t, Tto>(image, buffer, onehot, cf);
                case CV_16U: return onehot_helper<uint16_t, Tto>(image, buffer, onehot, cf);
                case CV_16S: return onehot_helper<int16_t, Tto>(image, buffer, onehot, cf);
                case CV_32S: return onehot_helper<int32_t, Tto>(image, buffer, onehot, cf);
                case CV_32F: return onehot_helper<float, Tto>(image, buffer, onehot, cf);
                case CV_64F: return onehot_helper<double, Tto>(image, buffer, onehot, cf);
            }
            CHECK(0) << "Mat type not supported.";
            return nullptr;
        }
    }

    // this is the main interface for most of the
    // deep learning libraries.
    class BatchImageStream: public ImageStream {
    public:
        enum {
            TASK_REGRESSION = 1,
            TASK_CLASSIFICATION = 2,
            TASK_PIXEL_REGRESSION = 3,
            TASK_PIXEL_CLASSIFICATION = 4
        };
    protected:
        cv::Scalar label_mean;//(0,0,0,0);
        cv::Scalar mean;
        unsigned onehot;
        unsigned batch;
        bool pad;
        bool bgr2rgb;
        int task;
    public:
        struct Config: public ImageStream::Config {
            float mean_color1;
            float mean_color2;
            float mean_color3;
            unsigned onehot;
            unsigned batch;
            bool pad;
            bool bgr2rgb;
            Config ():
                mean_color1(0),
                mean_color2(0),
                mean_color3(0),
                onehot(0), batch(1), pad(false), bgr2rgb(false) {
            }
        };

        BatchImageStream (fs::path const &path, Config const &c)
            : ImageStream(path, c),
            label_mean(0,0,0,0),
            mean(cv::Scalar(c.mean_color1, c.mean_color2, c.mean_color3)),
            onehot(c.onehot),
            batch(c.batch), pad(c.pad), bgr2rgb(c.bgr2rgb) {
            ImageStream::Value &v(ImageStream::peek());
            if (!v.annotation.data) {
                if (onehot > 0) {
                    task = TASK_CLASSIFICATION;
                }
                else {
                    task = TASK_REGRESSION;
                }
            }
            else {
                if (onehot) {
                    CHECK(v.annotation.channels() == 1);
                    task = TASK_PIXEL_CLASSIFICATION;
                }
                else {
                    task = TASK_PIXEL_REGRESSION;
                    if (c.annotate == "auto") {
                        label_mean = mean;
                    }
                }
            }
        }

        template <typename T=unsigned>
        void next_shape (vector<T> *images_shape,
                         vector<T> *labels_shape) {
            Value &next = ImageStream::peek();
            images_shape->clear();
            images_shape->push_back(batch);
            if (order == ORDER_NCHW) {
                images_shape->push_back(next.image.channels());
                images_shape->push_back(next.image.rows);
                images_shape->push_back(next.image.cols);
            }
            else if (order == ORDER_NHWC) {
                images_shape->push_back(next.image.rows);
                images_shape->push_back(next.image.cols);
                images_shape->push_back(next.image.channels());
            }
            else CHECK(0);

            labels_shape->clear();
            labels_shape->push_back(batch);
            switch (task) {
                case TASK_REGRESSION:
                    CHECK(!next.annotation.data);
                    break;
                case TASK_CLASSIFICATION:
                    CHECK(!next.annotation.data);
                    labels_shape->push_back(onehot); break;
                case TASK_PIXEL_REGRESSION:
                    CHECK(next.annotation.data);
                    if (order == ORDER_NCHW) {
                        labels_shape->push_back(next.annotation.channels());
                        labels_shape->push_back(next.annotation.rows);
                        labels_shape->push_back(next.annotation.cols);
                    }
                    else {
                        labels_shape->push_back(next.annotation.rows);
                        labels_shape->push_back(next.annotation.cols);
                        labels_shape->push_back(next.annotation.channels());
                    }
                    break;
                case TASK_PIXEL_CLASSIFICATION:
                    CHECK(next.annotation.data);
                    CHECK(next.annotation.channels() == 1);
                    if (order == ORDER_NCHW) {
                        labels_shape->push_back(onehot);
                        labels_shape->push_back(next.annotation.rows);
                        labels_shape->push_back(next.annotation.cols);
                    }
                    else {
                        labels_shape->push_back(next.annotation.rows);
                        labels_shape->push_back(next.annotation.cols);
                        labels_shape->push_back(onehot);
                    }
                    break;
                default: CHECK(0);
            }
        }

        template <typename T1=float, typename T2=float>
        void next_fill (T1 *images, T2 *labels, unsigned *npad = nullptr) {
            vector<unsigned> ishape, lshape;
            vector<unsigned> ishape2, lshape2;
            unsigned loaded = 0;
            try {
                for (unsigned i = 0; i < batch; ++i) {
                    if (i) {
                        next_shape(&ishape2, &lshape2);
                        CHECK(ishape == ishape2);
                        CHECK(lshape == lshape2);
                    }
                    else {
                        next_shape(&ishape, &lshape);
                    }
                    Value v(next());
                    if (order == ORDER_NCHW) {
                        images = impl::split_copy<T1>(v.image, images, mean, bgr2rgb);
                    }
                    else {
                        images = impl::copy<T1>(v.image, images, mean, bgr2rgb);
                    }
                    switch (task) {
                        case TASK_REGRESSION:
                            *labels = v.label;
                            ++labels;
                            break;
                        case TASK_CLASSIFICATION:
                            {
                                unsigned c = unsigned(v.label);
                                CHECK(c == v.label) << "float label for classification";
                                CHECK(c <= MAX_CATEGORIES);
                                std::fill(labels, labels + onehot, 0);
                                labels[c] = 1;
                                labels += onehot;
                            }
                            break;
                        case TASK_PIXEL_REGRESSION:
                            if (order == ORDER_NCHW) {
                                labels = impl::split_copy<T2>(v.annotation, labels, label_mean, bgr2rgb);
                            }
                            else {
                                labels = impl::copy<T2>(v.annotation, labels, label_mean, bgr2rgb);
                            }
                            break;
                        case TASK_PIXEL_CLASSIFICATION:
                            labels = impl::onehot_encode<T2>(v.annotation, labels, onehot, order == ORDER_NCHW);
                            break;
                        default: CHECK(0);
                    }
                    ++loaded;
                }
            }
            catch (EoS const &) {
            }
            //if ((pad && (loaded == 0)) || ((!pad) && (loaded < batch))) throw EoS();
            if (loaded == 0) throw EoS();
            if (npad) *npad = batch - loaded;
        }
    };
#endif

    cv::Mat decode_buffer (const_buffer, int mode = -1);
    void encode_raw (cv::Mat, string *);
    cv::Mat decode_raw (char const *, size_t);

    class ImageEncoder {
    protected:
        string code;
        vector<int> _params;
    public:
        ImageEncoder (string const &code_ =  string()): code(code_) {
        }
        vector<int> &params() { return _params; }
        void encode (cv::Mat const &image, string *);
    };

    class ImageReader: public ImageEncoder {
        int max;
        int resize;
        int mode;
    public:
        ImageReader (int max_ = 800, int resize_ = -1, int mode_ = cv::IMREAD_UNCHANGED, string const &code_ = string())
            : ImageEncoder(code_), max(max_), resize(resize_), mode(mode_) {
        }

        void read (fs::path const &path, string *data);
        void transcode (string const &input, string *data);
    };

    float LimitSize (cv::Mat input, int min_size, int max_size, cv::Mat *output);

    static inline float LimitSize (cv::Mat input, int max_size, cv::Mat *output) {
        return LimitSize(input, -1, max_size, output);
    }

    static inline cv::Point round (cv::Point2f p) {
        return cv::Point(std::round(p.x), std::round(p.y));
    }

}

