#pragma once
#include <random>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "picpac.h"

#define PICPAC_CONFIG picpac::BatchImageStream::Config

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
    PICPAC_CONFIG_UPDATE(C,min_size);\
    PICPAC_CONFIG_UPDATE(C,max_size);\
    PICPAC_CONFIG_UPDATE(C,resize_width);\
    PICPAC_CONFIG_UPDATE(C,resize_height);\
    PICPAC_CONFIG_UPDATE(C,crop_width);\
    PICPAC_CONFIG_UPDATE(C,crop_height);\
    PICPAC_CONFIG_UPDATE(C,round_div);\
    PICPAC_CONFIG_UPDATE(C,round_mod);\
    PICPAC_CONFIG_UPDATE(C,decode_mode);\
    PICPAC_CONFIG_UPDATE(C,annotate);\
    PICPAC_CONFIG_UPDATE(C,anno_type);\
    PICPAC_CONFIG_UPDATE(C,anno_copy);\
    PICPAC_CONFIG_UPDATE(C,anno_number);\
    PICPAC_CONFIG_UPDATE(C,anno_palette);\
    PICPAC_CONFIG_UPDATE(C,anno_color1); \
    PICPAC_CONFIG_UPDATE(C,anno_color2); \
    PICPAC_CONFIG_UPDATE(C,anno_color3); \
    PICPAC_CONFIG_UPDATE(C,anno_thickness);\
    PICPAC_CONFIG_UPDATE(C,anno_min_ratio); \
    PICPAC_CONFIG_UPDATE(C,perturb);\
    PICPAC_CONFIG_UPDATE(C,pert_colorspace); \
    PICPAC_CONFIG_UPDATE(C,pert_color1); \
    PICPAC_CONFIG_UPDATE(C,pert_color2); \
    PICPAC_CONFIG_UPDATE(C,pert_color3); \
    PICPAC_CONFIG_UPDATE(C,pert_angle); \
    PICPAC_CONFIG_UPDATE(C,pert_min_scale); \
    PICPAC_CONFIG_UPDATE(C,pert_max_scale); \
    PICPAC_CONFIG_UPDATE(C,pert_hflip); \
    PICPAC_CONFIG_UPDATE(C,pert_vflip); \
    PICPAC_CONFIG_UPDATE(C,mean_color1); \
    PICPAC_CONFIG_UPDATE(C,mean_color2); \
    PICPAC_CONFIG_UPDATE(C,mean_color3); \
    PICPAC_CONFIG_UPDATE(C,onehot);\
    PICPAC_CONFIG_UPDATE(C,batch);\
    PICPAC_CONFIG_UPDATE(C,pad);\
    PICPAC_CONFIG_UPDATE(C,bgr2rgb);\
    PICPAC_CONFIG_UPDATE(C,order);\
    PICPAC_CONFIG_UPDATE(C,point_radius);\
    PICPAC_CONFIG_UPDATE(C,multi_images);


namespace json11 {
    class Json;
}

namespace picpac {

    class ImageLoader {
    public:
        enum {
            ANNOTATE_NONE = 0,
            ANNOTATE_IMAGE = 1,
            ANNOTATE_JSON = 2,
            ANNOTATE_AUTO = 3   // for autoencoder, use input as annotation
        };
        enum {
            ANNOTATE_PALETTE_NONE = 0,
            ANNOTATE_PALETTE_TABLEAU20 = 1,
            ANNOTATE_PALETTE_TABLEAU20A = 2
        };
        enum {
            COLOR_DEFAULT = 0,
            COLOR_HSV = 1,
            COLOR_Lab = 2,
            COLOR_SAME = 3
            
        };
        enum {
            ORDER_NHWC = 1,
            ORDER_NCHW = 2,
            ORDER_DEFAULT = ORDER_NCHW
        };
        struct Config {
            int channels;   // -1: unchanged
            int min_size;
            int max_size;
            int resize_width;
            int resize_height;
            int crop_width;
            int crop_height;
            int round_div;
            int round_mod;
            int decode_mode;       // image load mode
            string annotate;
            int anno_type;  // annotate image opencv type
            bool anno_copy; // copy input image first for visualization
            bool anno_number;   // display number along the annotation for inspection
            string anno_palette;
            float anno_color1;
            float anno_color2;
            float anno_color3;
            int anno_thickness;
            float anno_min_ratio;
                            // -1 to fill (opencv rule)
            // perturbation
            bool perturb;
            // perturbation output retains input image size
            string pert_colorspace;
            string order;
            float pert_color1;
            float pert_color2;
            float pert_color3;
                            // perturb color range 
            float pert_angle;
                            // perturb angle range
            float pert_min_scale;
            float pert_max_scale;
            bool pert_hflip, pert_vflip;
            float pert_border;

            float point_radius; // in pixels

            int multi_images;   // for MultiImageLoader
            Config ()
                : channels(0),
                min_size(-1),
                max_size(-1),
                resize_width(-1),
                resize_height(-1),
                crop_width(-1),
                crop_height(-1),
                round_div(0),
                round_mod(0),
                decode_mode(cv::IMREAD_UNCHANGED),
                anno_type(CV_8UC1),
                anno_copy(false),
                anno_number(false),
                anno_color1(1),
                anno_color2(0),
                anno_color3(0),
                anno_thickness(CV_FILLED),
                anno_min_ratio(0),
                perturb(false),
                pert_color1(0),
                pert_color2(0),
                pert_color3(0),
                pert_angle(0),
                pert_min_scale(1),
                pert_max_scale(1),
                pert_hflip(false),
                pert_vflip(false),
                pert_border(cv::BORDER_CONSTANT),
                point_radius(3),
                multi_images(1)
            {
            }
        };

        struct Value {
            float label;
            cv::Mat image;
            cv::Mat annotation;
            void swap (Value &v) {
                std::swap(label, v.label);
                cv::swap(image, v.image);
                cv::swap(annotation, v.annotation);
            }
        };

        typedef Value CacheValue;

        struct PerturbVector {
            cv::Scalar color;
            float angle, scale;
            bool hflip, vflip;
            int shiftx, shifty;
        };

        ImageLoader (Config const &c)
            : config(c),
            annotate(ANNOTATE_NONE),
            delta_color1(-c.pert_color1, c.pert_color1),
            delta_color2(-c.pert_color2, c.pert_color2),
            delta_color3(-c.pert_color3, c.pert_color3),
            linear_angle(-c.pert_angle, c.pert_angle),
            linear_scale(c.pert_min_scale, c.pert_max_scale)
        {
            if (config.annotate == "json") {
                annotate = ANNOTATE_JSON;
            }
            else if (config.annotate == "image") {
                annotate = ANNOTATE_IMAGE;
            }
            else if (config.annotate == "auto") {
                annotate = ANNOTATE_AUTO;
            }
            if (config.anno_palette == "default"
                    || config.anno_palette == "tableau20") {
                anno_palette = ANNOTATE_PALETTE_TABLEAU20;
            }
            else if (config.anno_palette == "tableau20a") {
                anno_palette = ANNOTATE_PALETTE_TABLEAU20A;
            }
            else {
                anno_palette = ANNOTATE_PALETTE_NONE;
            }

            if (config.pert_colorspace == "Lab") {
                colorspace = COLOR_Lab;
            }
            else if (config.pert_colorspace == "HSV") {
                colorspace = COLOR_HSV;
            }
            else if (config.pert_colorspace == "SAME") {
                colorspace = COLOR_SAME;
            }
            else {
                colorspace = COLOR_DEFAULT;
            }
            if ((config.crop_width > 0) || (config.crop_height > 0)) {
                CHECK((config.crop_width > 0) && (config.crop_height > 0));
            }
            if (config.order == "NHWC") {
                order = ORDER_NHWC;
            }
            else if (config.order == "NCHW") {
                order = ORDER_NCHW;
            }
            else {
                order = ORDER_DEFAULT;
            }
            std::cerr << "ORDER: " << config.order << " " << order << std::endl;
        }

        template <typename RNG>
        void sample (RNG &e, PerturbVector *p) {
            if (config.perturb) {
                p->hflip = bool(e() % 2) && config.pert_hflip;
                p->vflip = bool(e() % 2) && config.pert_vflip;
                p->color[0] = delta_color1(e);
                p->color[1] = delta_color2(e);
                p->color[2] = delta_color3(e);
                p->angle = linear_angle(e);
                p->scale = linear_scale(e);
                p->shiftx = e();
                p->shifty = e();
            }
        }
        // helper functions

        // loads image from buffer, apply preprocessing transformations but do not perturb
        struct LoadState {
            cv::Size size;
            cv::Mat copy_for_anno;
            bool crop;
            cv::Rect crop_bb;
            LoadState (): crop(false) {
            }
        };
        cv::Mat preload_image (const_buffer buffer, LoadState *state) const;
        cv::Mat preload_annotation (const_buffer buffer, LoadState *state) const;
        cv::Mat process_image (cv::Mat image, PerturbVector const &pv, LoadState const *state, bool is_anno) const;

        cv::Mat load_image (const_buffer buffer, PerturbVector const &pv, LoadState *state) const;
        cv::Mat load_annotation (const_buffer buffer, PerturbVector const &pv, LoadState *state) const;

        void load (RecordReader, PerturbVector const &, Value *,
                CacheValue *c = nullptr, std::mutex *m = nullptr) const;

    protected:
        Config config;
        int annotate;
        int anno_palette;
        int colorspace;
        int order;
        std::uniform_int_distribution<int> delta_color1; //(min_R, max_R);
        std::uniform_int_distribution<int> delta_color2; //(min_R, max_R);
        std::uniform_int_distribution<int> delta_color3; //(min_R, max_R);
        std::uniform_real_distribution<float> linear_angle;
        std::uniform_real_distribution<float> linear_scale;
    };

    typedef PrefetchStream<ImageLoader> ImageStream;

    class MultiImageLoader: public ImageLoader {
    public:
        using ImageLoader::ImageLoader;

        struct Value {
            float label;
            vector<cv::Mat> images;
            cv::Mat annotation;

            void swap (Value &v) {
                std::swap(label, v.label);
                images.swap(v.images);
                cv::swap(annotation, v.annotation);
            }
        };

        void load (RecordReader reader, PerturbVector const &pv, Value *value,
                CacheValue *c = nullptr, std::mutex *m = nullptr) const {
            CHECK(c == nullptr);
            CHECK(m == nullptr);
            Record r;
            reader(&r); // disk access
            value->label = r.meta().label;
            LoadState state;
            for (int i = 0; i < config.multi_images; ++i) {
                cv::Mat image = preload_image(r.field(i), &state);
                image = process_image(image, pv, &state, false);
                value->images.push_back(image);
            }
            if (annotate != ANNOTATE_NONE) {
                cv::Mat image = preload_annotation(r.field(config.multi_images), &state);
                value->annotation = process_image(image, pv, &state, true);
            }
        }
    };

    typedef PrefetchStream<MultiImageLoader> MultiImageStream;

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

    class Shape {
        string _type;
        bool _have_label;
        cv::Scalar _label;
    public:
        Shape (char const *t): _type(t), _have_label(false), _label(0,0,0) {}
        virtual ~Shape () {}
        virtual void draw (cv::Mat *, cv::Scalar v, int thickness = CV_FILLED) const = 0;
        virtual void bbox (cv::Rect_<float> *) const = 0;
        virtual void zoom (cv::Rect_<float> const &) = 0;
        virtual void dump (json11::Json *) const = 0;
        string const &type () const {
            return _type;
        }
        void setLabel (cv::Scalar l) {
            _have_label = true;
            _label = l;
        }
        bool haveLabel () const { return _have_label; }
        cv::Scalar label () const { return _label; }
        virtual std::shared_ptr<Shape> clone () const = 0;
        static std::shared_ptr<Shape> create (json11::Json const &geo, cv::Size, ImageLoader::Config const &config);
    };

    struct Annotation {
        vector<std::shared_ptr<Shape>> shapes;
        Annotation () {}
        Annotation (string const &txt, cv::Size, ImageLoader::Config const &config);
        void dump (string *) const;
        void draw (cv::Mat *m, cv::Scalar v, int thickness = -1, vector<cv::Scalar> const *palette=nullptr, bool show_number = false) const;
        void bbox (cv::Rect_<float> *bb) const;
        void zoom (cv::Rect_<float> const &bb);
    };
}

