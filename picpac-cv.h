#pragma once
#include <random>
#include <opencv2/opencv.hpp>
#include "picpac.h"

namespace picpac {

    class ImageLoader {
    public:
        enum {
            ANNOTATE_NONE = 0,
            ANNOTATE_IMAGE = 1,
            ANNOTATE_JSON = 2
        };
        struct Config {
            int channels;
            cv::Size resize;
            int mode;
            int annotate;
            int anno_type;
            bool anno_copy;
            cv::Scalar anno_color;
            int anno_thickness;
            bool perturb;
            cv::Scalar pert_color;
            float pert_angle;
            float pert_min_scale;
            float pert_max_scale;
            bool pert_hflip, pert_vflip;
            Config ()
                : channels(0),
                resize(0, 0), // do not resize by default
                mode(cv::IMREAD_UNCHANGED),
                annotate(ANNOTATE_NONE),
                anno_type(CV_8UC1),
                anno_copy(false),
                anno_color(1),
                anno_thickness(CV_FILLED),
                perturb(false),
                pert_color(0,0,0),
                pert_angle(0),
                pert_min_scale(1),
                pert_max_scale(1),
                pert_hflip(false),
                pert_vflip(false) {
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
        };

        ImageLoader (Config const &c)
            : config(c),
            delta_color1(-c.pert_color[0], c.pert_color[0]),
            delta_color2(-c.pert_color[1], c.pert_color[1]),
            delta_color3(-c.pert_color[2], c.pert_color[2]),
            linear_angle(-c.pert_angle, c.pert_angle),
            linear_scale(c.pert_min_scale, c.pert_max_scale)
        {
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
            }
        }

        void load (RecordReader, PerturbVector const &, Value *,
                CacheValue *c = nullptr, std::mutex *m = nullptr) const;

    private:
        Config config;
        std::uniform_int_distribution<int> delta_color1; //(min_R, max_R);
        std::uniform_int_distribution<int> delta_color2; //(min_R, max_R);
        std::uniform_int_distribution<int> delta_color3; //(min_R, max_R);
        std::uniform_real_distribution<float> linear_angle;
        std::uniform_real_distribution<float> linear_scale;
    };

    typedef PrefetchStream<ImageLoader> ImageStream;

    namespace impl {
        template <typename Tfrom = uint8_t, typename Tto = float>
        Tto *split_helper (cv::Mat image, Tto *buffer, bool bgr2rgb) {
            Tto *ptr_b = buffer;
            Tto *ptr_g = buffer;
            Tto *ptr_r = buffer;
            if (bgr2rgb) {
                ptr_g += image.total();
                ptr_b += 2 * image.total();
            }
            else if (image.channels() > 1) {
                ptr_g += image.total();
                ptr_r += 2 * image.total();
            }
            unsigned off = 0;
            for (int i = 0; i < image.rows; ++i) {
                Tfrom const *line = image.ptr<Tfrom const>(i);
                for (int j = 0; j < image.cols; ++j) {
                    if (image.channels() > 1) {
                        ptr_b[off] = *line++;
                        ptr_g[off] = *line++;
                    }
                    ptr_r[off] = *line++;
                    ++off;
                }
            }
            CHECK(off == image.total());
            if (image.channels() == 1) return buffer + image.total();
            return buffer + 3 * image.total();
        }

        template <typename Tto = float>
        Tto *split_copy (cv::Mat image, Tto *buffer, bool bgr2rgb) {
            int depth = image.depth();
            int ch = image.channels();
            CHECK(ch == 1 || ch == 3);
            switch (depth) {
                case CV_8U: return split_helper<uint8_t, Tto>(image, buffer, bgr2rgb);
                case CV_8S: return split_helper<int8_t, Tto>(image, buffer, bgr2rgb);
                case CV_16U: return split_helper<uint16_t, Tto>(image, buffer, bgr2rgb);
                case CV_16S: return split_helper<int16_t, Tto>(image, buffer, bgr2rgb);
                case CV_32S: return split_helper<int32_t, Tto>(image, buffer, bgr2rgb);
                case CV_32F: return split_helper<float, Tto>(image, buffer, bgr2rgb);
                case CV_64F: return split_helper<double, Tto>(image, buffer, bgr2rgb);
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
        unsigned onehot;
        unsigned batch;
        bool pad;
        bool bgr2rgb;
        int task;

    public:
        struct Config: public ImageStream::Config {
            unsigned onehot;
            unsigned batch;
            bool pad;
            bool bgr2rgb;
            Config (): onehot(0), batch(1), pad(false), bgr2rgb(false) {
            }
        };

        BatchImageStream (fs::path const &path, Config const &c)
            : ImageStream(path, c), onehot(c.onehot),
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
                    images = impl::split_copy<T1>(v.image, images, bgr2rgb);
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
                            labels = impl::split_copy<T2>(v.annotation, labels, bgr2rgb);
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
}

