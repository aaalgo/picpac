#pragma once
#include <random>
#include <opencv2/opencv.hpp>
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
    PICPAC_CONFIG_UPDATE(C,decode_mode);\
    PICPAC_CONFIG_UPDATE(C,annotate);\
    PICPAC_CONFIG_UPDATE(C,anno_type);\
    PICPAC_CONFIG_UPDATE(C,anno_copy);\
    PICPAC_CONFIG_UPDATE(C,anno_color1); \
    PICPAC_CONFIG_UPDATE(C,anno_color2); \
    PICPAC_CONFIG_UPDATE(C,anno_color3); \
    PICPAC_CONFIG_UPDATE(C,anno_thickness);\
    PICPAC_CONFIG_UPDATE(C,anno_min_ratio); \
    PICPAC_CONFIG_UPDATE(C,perturb);\
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
    PICPAC_CONFIG_UPDATE(C,bgr2rgb);

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
        struct Config {
            int channels;   // -1: unchanged
            int min_size;
            int max_size;
            int resize_width;
            int resize_height;
            int crop_width;
            int crop_height;
            int decode_mode;       // image load mode
            string annotate;
            int anno_type;  // annotate image opencv type
            bool anno_copy; // copy input image first for visualization
            float anno_color1;
            float anno_color2;
            float anno_color3;
            int anno_thickness;
            float anno_min_ratio;
                            // -1 to fill (opencv rule)
            // perturbation
            bool perturb;
            // perturbation output retains input image size
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


            Config ()
                : channels(0),
                min_size(-1),
                max_size(-1),
                resize_width(-1),
                resize_height(-1),
                crop_width(-1),
                crop_height(-1),
                decode_mode(cv::IMREAD_UNCHANGED),
                anno_type(CV_8UC1),
                anno_copy(false),
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
                pert_border(cv::BORDER_CONSTANT)
            {
            }
        };

        struct Value {
            float label;
            cv::Mat image;
            cv::Mat annotation;
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

        void load (RecordReader, PerturbVector const &, Value *,
                CacheValue *c = nullptr, std::mutex *m = nullptr) const;

    private:
        Config config;
        int annotate;
        std::uniform_int_distribution<int> delta_color1; //(min_R, max_R);
        std::uniform_int_distribution<int> delta_color2; //(min_R, max_R);
        std::uniform_int_distribution<int> delta_color3; //(min_R, max_R);
        std::uniform_real_distribution<float> linear_angle;
        std::uniform_real_distribution<float> linear_scale;
    };

    typedef PrefetchStream<ImageLoader> ImageStream;

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

        
        template <typename Tfrom, typename Tto = float>
        Tto *onehot_helper (cv::Mat image, Tto *buffer, unsigned onehot) {
            size_t ch_size = image.total();
            size_t total_size = ch_size * onehot;
            Tto *buffer_end = buffer + total_size;
            std::fill(buffer, buffer_end, 0);
            int off = 0;
            for (int i = 0; i < image.rows; ++i) {
                Tfrom const *row = image.ptr<Tfrom const>(i);
                for (int j = 0; j < image.cols; ++j) {
                    Tfrom v = row[j];
                    unsigned c(v);
                    CHECK(c == v);
                    CHECK(c <= MAX_CATEGORIES);
                    Tto *plane = buffer + c * ch_size;
                    plane[off] = 1;
                    ++off;
                }
            }
            return buffer_end;
        }

        template <typename Tto = float>
        Tto *onehot_encode (cv::Mat image, Tto *buffer, unsigned onehot) {
            int depth = image.depth();
            int ch = image.channels();
            CHECK(ch == 1);
            switch (depth) {
                case CV_8U: return onehot_helper<uint8_t, Tto>(image, buffer, onehot);
                case CV_8S: return onehot_helper<int8_t, Tto>(image, buffer, onehot);
                case CV_16U: return onehot_helper<uint16_t, Tto>(image, buffer, onehot);
                case CV_16S: return onehot_helper<int16_t, Tto>(image, buffer, onehot);
                case CV_32S: return onehot_helper<int32_t, Tto>(image, buffer, onehot);
                case CV_32F: return onehot_helper<float, Tto>(image, buffer, onehot);
                case CV_64F: return onehot_helper<double, Tto>(image, buffer, onehot);
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
    private:
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
            images_shape->push_back(next.image.channels());
            images_shape->push_back(next.image.rows);
            images_shape->push_back(next.image.cols);

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
                    labels_shape->push_back(next.annotation.channels());
                    labels_shape->push_back(next.annotation.rows);
                    labels_shape->push_back(next.annotation.cols);
                    break;
                case TASK_PIXEL_CLASSIFICATION:
                    CHECK(next.annotation.data);
                    CHECK(next.annotation.channels() == 1);
                    labels_shape->push_back(onehot);
                    labels_shape->push_back(next.annotation.rows);
                    labels_shape->push_back(next.annotation.cols);
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
                    images = impl::split_copy<T1>(v.image, images, mean, bgr2rgb);
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
                            labels = impl::split_copy<T2>(v.annotation, labels, label_mean, bgr2rgb);
                            break;
                        case TASK_PIXEL_CLASSIFICATION:
                            labels = impl::onehot_encode<T2>(v.annotation, labels, onehot);
                            break;
                        default: CHECK(0);
                    }
                    ++loaded;
                }
            }
            catch (EoS const &) {
            }
            if ((pad && (loaded == 0)) || ((!pad) && (loaded < batch))) throw EoS();
            if (npad) *npad = batch - loaded;
        }
    };

    cv::Mat decode_buffer (const_buffer, int mode = -1);
    void encode_raw (cv::Mat, string *);
    cv::Mat decode_raw (char const *, size_t);

    class ImageEncoder {
    protected:
        string code;
    public:
        ImageEncoder (string const &code_ =  string()): code(code_) {
        }
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
    };

    float LimitSize (cv::Mat input, int min_size, int max_size, cv::Mat *output);

    static inline float LimitSize (cv::Mat input, int max_size, cv::Mat *output) {
        return LimitSize(input, -1, max_size, output);
    }

    class Shape {
        string _type;
    public:
        Shape (char const *t): _type(t) {}
        virtual ~Shape () {}
        virtual void draw (cv::Mat *, cv::Scalar v, int thickness = CV_FILLED) const = 0;
        virtual void bbox (cv::Rect_<float> *) const = 0;
        virtual void zoom (cv::Rect_<float> const &) = 0;
        virtual void dump (json11::Json *) const = 0;
        string const &type () const {
            return _type;
        }
        virtual std::shared_ptr<Shape> clone () const = 0;
        static std::shared_ptr<Shape> create (json11::Json const &geo);
    };

    struct Annotation {
        vector<std::shared_ptr<Shape>> shapes;
        Annotation () {}
        Annotation (string const &txt);
        void dump (string *) const;
        void draw (cv::Mat *m, cv::Scalar v, int thickness = -1) const;
        void bbox (cv::Rect_<float> *bb) const;
        void zoom (cv::Rect_<float> const &bb);
    };
}

