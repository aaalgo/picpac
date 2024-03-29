#include <iostream>
#include "picpac-image.h"

namespace picpac {

    class Drop: public Transform {
        int index;
    public:
        Drop (json const &spec) {
            index = spec.value<int>("index", 1);
        }

        virtual size_t apply (Sample *sample, bool, void const *) const {
            auto &facet = sample->facets[index];
            facet.type = Facet::NONE;
            return 0;
        }
    };

    class Featurize: public Transform {
        int index;
    public:
        Featurize (json const &spec) {
            index = spec.value<int>("index", 1);
        }

        virtual size_t apply (Sample *sample, bool, void const *) const {
            auto &facet = sample->facets[index];
            facet.type = Facet::FEATURE;
            return 0;
        }
    };


    class BorderConfig {
    protected:
        int border_type;
        cv::Scalar border_value;
    public:
        BorderConfig (json const &spec, int type, cv::Scalar value)
            : border_type(type), border_value(value)
        {
            string b = spec.value<string>("border_type", string("constant"));
            if (b == "replicate") {
                border_type = cv::BORDER_REPLICATE;
            }
            else if (b == "constant") {
                border_type = cv::BORDER_CONSTANT;
            }
            else if (b == "reflect") {
                border_type = cv::BORDER_REFLECT_101;
            }
            else CHECK(0); // << "Border type " << b << " not recognized";
            // TODO: config border value
        }
    };

    class Mask: public Transform {
        int ref;
        int index;
        int value;
    public:
        Mask (json const &spec) {
            ref = spec.value<int>("ref", 0);
            index = spec.value<int>("index", -1);
            value = spec.value<int>("value", 1);
        }
        virtual size_t apply (Sample *sample, bool, void const *) const {
            //auto &ref_image = sample->facets[ref].image;
            int idx = index;
            if (idx < 0) {  // add image
                sample->facets.emplace_back();
                idx = sample->facets.size()-1;
            }
            auto &facet = sample->facets[idx];
            facet.type = Facet::LABEL;
            cv::Size sz = sample->facets[ref].check_size();
            /*
            if (!ref_image.data) {
                return 0;
            }
            */
            facet.image = cv::Mat(sz, CV_8U, cv::Scalar(value));
            return 0;
        }
    };

    class ErodeMask: public Transform, public BorderConfig {
        int index;
        cv::Mat kernel;
        int border_value;
        int iterations;
    public:
        ErodeMask (json const &spec): BorderConfig(spec, cv::BORDER_CONSTANT, cv::Scalar(1,1,1,1)) {
            index = spec.value<int>("index", -1);
            int r = spec.value<int>("radius", 0);
            iterations = spec.value<int>("iterations", 1);
            kernel = cv::getStructuringElement(cv::MORPH_RECT,
                            cv::Size(2*r+1,2*r+1),
                            cv::Point(r,r));
        }
        virtual size_t apply (Sample *sample, bool, void const *) const {
            int idx = index;
            if (idx < 0) {  // add image
                idx = sample->facets.size()-1;
            }
            auto &facet = sample->facets[idx];
            CHECK(facet.type = Facet::LABEL);
            if (!facet.image.data) {
                return 0;
            }
            CHECK(facet.image.type() == CV_8UC1);
            cv::erode(facet.image,
                      facet.image, 
                      kernel,
                      cv::Point(-1, -1),
                      iterations,
                      border_type, border_value);
            return 0;
        }
    };

    class Rasterize: public Transform {
        int copy;
        int index;
        int type;
        RenderOptions opt;
    public:
        Rasterize (json const &spec) {
            copy = spec.value<int>("copy", -1);
            index = spec.value<int>("index", 1);
            int channels = spec.value<int>("channels", 1);
            int dtype = dtype_np2cv(spec.value<string>("dtype", string("uint8")));
            type = CV_MAKETYPE(dtype, channels);
            opt.use_palette = spec.value<bool>("use_palette", bool(opt.use_palette));
            opt.use_tag = spec.value<bool>("use_tag", bool(opt.use_tag));
            opt.use_serial = spec.value<bool>("use_serial", bool(opt.use_serial));
            opt.thickness = spec.value<int>("thickness", int(opt.thickness));
            CHECK(((!opt.use_palette) || (!opt.use_tag))); // << "Cannot use both tag and palette";
        }
        virtual size_t apply (Sample *sample, bool, void const *) const {
            auto &facet = sample->facets[index];
            if (copy >= 0) {
                facet.image = sample->facets[copy].image.clone();
            }
            else {
                if (facet.image.data) {
                    ;
                }
                else {
                    facet.image = cv::Mat(facet.annotation.size, type, cv::Scalar(0,0,0,0));
                }
                //std::cerr << facet.image.cols << 'x' << facet.image.rows << std::endl;
            }
            facet.annotation.render(&facet.image, opt);
            facet.annotation.clear();
            return 0;
        }
    };

    class Normalize: public Transform {
        bool do_mu;
        cv::Scalar mu;
        bool do_sigma;
        cv::Scalar sigma;
        void load_scalar (cv::Scalar &v, json const &j) {
            if (j.is_array()) {
                int o = 0;
                for (json const &e: j) {
                    v[o++] = e.get<float>();
                    if (o == 4) break;
                }
            }
            else {
                float a = j.get<float>();
                v = cv::Scalar(a,a,a,a);
            }
        }
    public:
        Normalize (json const &spec): do_mu(false), mu(0,0,0,0), do_sigma(false), sigma(1,1,1,1) {
            auto m = spec.find("mean");
            if (m != spec.end()) {
                do_mu = true;
                load_scalar(mu, *m);
            }
            auto s = spec.find("std");
            if (s != spec.end()) {
                do_sigma = true;
                load_scalar(sigma, *s);
            }
            /*
            std::cerr << do_mu;
            if (do_mu) {
                std::cerr << " " << mu[0] << ':' << mu[1] << ':' << mu[2] << ':' << mu[3];
            }
            std::cerr << do_sigma;
            if (do_sigma) {
                std::cerr << " " << sigma[0] << ':' << sigma[1] << ':' << sigma[2] << ':' << sigma[3];
            }
            std::cerr << std::endl;
            */
        }
        virtual size_t apply (Sample *sample, bool, void const *) const {
            do {
                if (sample->facets.size() == 1) {
                    break;
                }
                /*
                else if (sample->facets.size() == 2) {
                    if (sample->facets[1].type == Facet::LABEL) break;
                }
                CHECK(0) << "Normalize doens't support non-standard configuration.";
                */
            } while (false);
            auto &facet = sample->facets[0];
            if (do_mu) {
                facet.image -= mu;
            }
            if (do_sigma) {
                vector<cv::Mat> channels;
                cv::split(facet.image, channels);
                for (unsigned i = 0; i < channels.size(); ++i) {
                    channels[i] /= sigma[i];
                }
                cv::merge(channels, facet.image);
                //facet.image /= sigma;
            }
            return 0;
        }
    };

    class Resize: public Transform {
        // if size/width/height is provided, use the provided value
        // otherwise, limit size using min_size & max_size
        int size;
        int width, height;

        int min_size;
        int max_size;
    public:
        Resize (json const &spec)
            : size(spec.value<int>("size", 0)),
              width(spec.value<int>("width", int(size))),
              height(spec.value<int>("height", int(size))),
              min_size(spec.value<int>("min_size", 0)),
              max_size(spec.value<int>("max_size", numeric_limits<int>::max()))
        {
            if (width > 0) { CHECK(height > 0); CHECK(width >= min_size && width <= max_size); }
            if (height > 0) { CHECK(width > 0); CHECK(height >= min_size && height <= max_size); }
        }

        virtual size_t apply_one (Facet *facet, bool, void const *buf) const {
            while (true) {
                //CHECK(facet->type != Facet::FEATURE);
                if (facet->type == Facet::FEATURE) break;
                if ((!facet->image.data) && (facet->annotation.empty())) break;
                cv::Size sz0;
                if (facet->image.data) {
                    sz0 = facet->image.size();
                }
                else {
                    sz0 = facet->annotation.size;
                }

                int w = 0, h = 0;
                if (width > 0) {
                    CHECK(height > 0);
                    w = width;
                    h = height;
                }
                else {  // check max
                    int min = std::min(sz0.width, sz0.height);
                    int max = std::max(sz0.width, sz0.height);
                    if (max > max_size) {
                        w = sz0.width * max_size / max;
                        h = sz0.height * max_size / max;
                    }
                    else if (min < min_size) {
                        w = sz0.width * min_size / min;
                        h = sz0.height * min_size / min;
                    }
                }
                if (w == 0 || h == 0) break;
                cv::Size sz(w, h);
                if (sz == sz0) break;

                if (facet->image.size() == sz) break;
                if (facet->image.data) {
                    cv::Mat tmp;
                    cv::resize(facet->image, tmp, sz, 0, 0, facet->type == Facet::LABEL ? cv::INTER_NEAREST: cv::INTER_LINEAR);
                    facet->image = tmp;
                }
                if (!facet->annotation.empty()) {
                    CHECK(facet->annotation.size == sz0);
                    facet->annotation.size = sz;
                    float fx = 1.0 * sz.width / sz0.width;
                    float fy = 1.0 * sz.height / sz0.height;
                    facet->annotation.transform([fx, fy](vector<cv::Point2f> *f) {
                            for (auto &pt: *f) {
                                pt.x *= fx;
                                pt.y *= fy;
                            }
                    });
                }
                break;
            }
            return 0;
        }
    };

    class Scale: public Transform {
        // if size/width/height is provided, use the provided value
        // otherwise, limit size using min_size & max_size
        float scale;
    public:
        Scale (json const &spec)
            : scale(spec.value<float>("scale", 1.0))
        {
        }

        virtual size_t apply_one (Facet *facet, bool, void const *buf) const {
            if (scale == 1.0) return 0;
            while (true) {
                //CHECK(facet->type != Facet::FEATURE);
                if (facet->type == Facet::FEATURE) break;
                if ((!facet->image.data) && (facet->annotation.empty())) break;
                cv::Size sz0;
                if (facet->image.data) {
                    sz0 = facet->image.size();
                }
                else {
                    sz0 = facet->annotation.size;
                }

                cv::Size sz(sz0.width * scale, sz0.height * scale);

                if (facet->image.data) {
                    cv::Mat tmp;
                    cv::resize(facet->image, tmp, sz, 0, 0, facet->type == Facet::LABEL ? cv::INTER_NEAREST: cv::INTER_LINEAR);
                    facet->image = tmp;
                }
                if (!facet->annotation.empty()) {
                    CHECK(facet->annotation.size == sz0);
                    facet->annotation.size = sz;
                    float sc = scale;
                    facet->annotation.transform([sc](vector<cv::Point2f> *f) {
                            for (auto &pt: *f) {
                                pt.x *= sc;
                                pt.y *= sc;
                            }
                    });
                }
                break;
            }
            return 0;
        }
    };


    class Pyramid: public Transform {
        int count;
        float rate;
    public:
        Pyramid (json const &spec)
            : count(spec.value<int>("count", 0)),
              rate(spec.value<float>("rate", 0.5))
        {
            CHECK(count >= 0);
            CHECK(rate > 0);
        }

        virtual size_t apply (Sample *sample, bool, void const *) const {
            auto facet = &sample->facets[0];
            CHECK(facet->type != Facet::FEATURE);
            CHECK(facet->type == Facet::IMAGE);
            if ((!facet->image.data) || facet->image.rows == 0 || facet->image.cols == 0) {
                for (int i = 0; i < count; ++i) {
                    sample->facets.emplace_back(cv::Mat());
                }
            }
            else {
                float fx = rate;
                for (int i = 0; i < count; ++i) {
                    auto facet = &sample->facets[0];
                    CHECK(fx > 0);
                    cv::Mat tmp;
                    cv::resize(facet->image, tmp, cv::Size(), fx, fx);
                    sample->facets.emplace_back(tmp);
                    fx *= rate;
                }
            }
            return 0;
        }
    };

    class Pad: public Transform {
        // does not randomize
        int size, width, height;
        struct PerturbVector {
        };
    public:
        Pad (json const &spec) {
            size = spec.value<int>("size", 400);
            width = spec.value<int>("width", int(size));
            height = spec.value<int>("height", int(size));
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, bool, void *buf) const {
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, bool, void const *buf) const {
            if (facet->type == Facet::FEATURE) return sizeof(PerturbVector);
            if (facet->image.data) {
                cv::Mat to(cv::Size(width, height), facet->image.type(), cv::Scalar(0,0,0,0));
                int w = facet->image.cols;
                int h = facet->image.rows;
                CHECK(w <= width && h <= height);
                facet->image.copyTo(to(cv::Rect(0, 0, w, h)));
                facet->image = to;
            }
            else if (!facet->annotation.empty()) {
                facet->annotation.size = cv::Size(width, height);
            }
            else CHECK(0);
            return sizeof(PerturbVector);
        }
    };

    class Clip: public Transform, public BorderConfig {
        int size, min, max, round;
        int width, height;
        int min_width, max_width;
        int min_height, max_height;
        int round_width, round_height;
        int max_crop;
        int max_crop_width;
        int max_crop_height;

        int max_shift;
        int max_shift_x, max_shift_y;

        static void adjust_crop_pad_range (int &from_x, int &from_width,
                                           int &to_x, int &to_width, bool perturb, int shiftx) {
            if (from_width < to_width) {
                int margin = to_width - from_width;
                if (from_x == 0) {
                    if (perturb) {
                        to_x += shiftx % margin;
                    }
                    else {
                        to_x += margin / 2;
                    }
                }
                to_width = from_width;
            }
            else if (from_width > to_width) {
                int margin = from_width - to_width;
                if (to_x == 0) {
                    if (perturb) {
                        from_x += shiftx % margin;
                    }
                    else {
                        from_x += margin / 2;
                    }
                }
                from_width = to_width;
            }
        }
        static void adjust_shift_pos (int &from_x, int &from_width, int from_size, int shift) {
            from_x = std::min(from_x + shift, from_size);
            if (from_x + from_width > from_size) {
                from_width = from_size - from_x;
            }
        }
        static void adjust_shift(int &from_x, int &from_width, int from_size,
                          int &to_x, int &to_width, int to_size, int shift) {
            if (shift > 0) {
                adjust_shift_pos(from_x, from_width, from_size, shift);
            }
            else {
                adjust_shift_pos(to_x, to_width, to_size, -shift);
            }
        }
    public:
        struct PerturbVector {
            int xrand;
            int yrand;
            int xcrop1, xcrop2;
            int ycrop1, ycrop2;
            int xshift;
            int yshift;
        };

        Clip (json const &spec)
            : BorderConfig(spec, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0)),
              size(spec.value<int>("size", 0)),
              min(spec.value<int>("min", 0)),
              max(spec.value<int>("max", 0)),
              round(spec.value<int>("round", 0)),
              width(spec.value<int>("width", int(size))),
              height(spec.value<int>("height", int(size))),
              min_width(spec.value<int>("min_width", int(min))),
              max_width(spec.value<int>("max_width", int(max))),
              min_height(spec.value<int>("min_height", int(min))),
              max_height(spec.value<int>("max_height", int(max))),
              round_width(spec.value<int>("round_width", int(round))),
              round_height(spec.value<int>("round_height", int(round))),
              max_crop(spec.value<int>("max_crop", 0)),
              max_crop_width(spec.value<int>("max_crop_width", int(max_crop))),
              max_crop_height(spec.value<int>("max_crop_height", int(max_crop))),
              max_shift(spec.value<int>("shift", 0)),
              max_shift_x(spec.value<int>("shift_x", int(max_shift))),
              max_shift_y(spec.value<int>("shift_y", int(max_shift)))

        {
            if ((width > 0) || (height > 0)) {
                CHECK((width > 0) && (height > 0));
                min_width = max_width = width;
                min_height = max_height = height;
                if (round_width > 0) {
                    CHECK(width % round_width == 0);
                }
                if (round_height > 0) {
                    CHECK(height % round_height == 0);
                }
            }
            CHECK(min_width <= max_width);
            CHECK(min_height <= max_height);
        }

        virtual size_t pv_size () const { return sizeof(PerturbVector); }

        virtual size_t pv_sample (random_engine &rng, bool perturb, void *buf) const {
            // TODO! perturb
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            if (perturb) {
                pv->xrand = rng();
                pv->yrand = rng();
                if (max_crop_width > 0) {
                    pv->xcrop1 = rng() % max_crop_width;
                    pv->xcrop2 = rng() % max_crop_width;
                }
                else {
                    pv->xcrop1 = pv->xcrop2 = 0;
                }
                if (max_crop_height > 0) {
                    pv->ycrop1 = rng() % max_crop_height;
                    pv->ycrop2 = rng() % max_crop_height;
                }
                else {
                    pv->ycrop1 = pv->ycrop2 = 0;
                }
                // -max_shift_x 0 max_shift_x
                pv->xshift = (rng() % (2 * max_shift_x + 1)) - max_shift_x;
                pv->yshift = (rng() % (2 * max_shift_y + 1)) - max_shift_y;
            }
            else {
                pv->xrand = pv->yrand = 0;
                pv->xcrop1 = pv->xcrop2 = 0;
                pv->ycrop1 = pv->ycrop2 = 0;
                pv->xshift = pv->yshift = 0;
            }
            return sizeof(PerturbVector);
        }

        static void update_crop (int size, int min, int max, int &crop1, int &crop2) {
            CHECK(size >= min);
            int above = size - min;
            int crop = crop1 + crop2;
            if (above < crop){
                int delta = crop - above;
                int a = delta / 2;
                int b = delta - a;
                if (a > crop1) {
                    a = crop1;
                    b = delta - a;
                }
                if (b > crop2) {
                    b = crop2;
                    a = delta - crop2;
                }
                CHECK(a + b == delta);
                crop1 -= a;
                crop2 -= b;
            }
            CHECK(crop1 >= 0);
            CHECK(crop2 >= 0);
            CHECK(size - crop1 - crop2 >= min);

        }

        virtual size_t apply_one (Facet *facet, bool perturb, void const *buf) const {
            // TODO perturb
            if (facet->type == Facet::FEATURE) return sizeof(PerturbVector);
            cv::Size sz0 = facet->check_size();
            if (sz0.width == 0) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            int xcrop1 = pv->xcrop1;
            int xcrop2 = pv->xcrop2;
            int ycrop1 = pv->ycrop1;
            int ycrop2 = pv->ycrop2;

            cv::Size sz = sz0;
            if (sz.width < min_width) sz.width = min_width;
            update_crop(sz.width, min_width, max_width, xcrop1, xcrop2);
            if (sz.width - xcrop1 - xcrop2 > max_width) sz.width = max_width + xcrop1 + xcrop2;
            if (sz.height < min_height) sz.height = min_height;
            update_crop(sz.height, min_height, max_height, ycrop1, ycrop2);
            if (sz.height - ycrop1 - ycrop2 > max_height) sz.height = max_height + ycrop1 + ycrop2;
            if (round_width > 0) {
                int mod = (sz.width - xcrop1 - xcrop2) % round_width;
                sz.width -= mod;
            }
            if (round_height > 0) {
                int mod = (sz.height - ycrop1 - ycrop2) % round_height;
                sz.height -= mod;
            }
            sz.width -= xcrop1 + xcrop2;
            sz.height -= ycrop1 + ycrop2;
            if (sz == sz0 && (pv->xshift == 0) && (pv->yshift == 0)) return sizeof(PerturbVector);


            int from_x = xcrop1, from_width = sz0.width - xcrop1 - xcrop2;
            int from_y = ycrop1, from_height = sz0.height - ycrop1 - ycrop2;


            int to_x = 0, to_width = sz.width;
            int to_y = 0, to_height = sz.height;

            adjust_shift(from_x, from_width, sz0.width, to_x, to_width, sz.width, pv->xshift);
            adjust_shift(from_y, from_height, sz0.height, to_y, to_height, sz.height, pv->yshift);

            adjust_crop_pad_range(from_x, from_width, to_x, to_width, true, pv->xrand);
            adjust_crop_pad_range(from_y, from_height, to_y, to_height, true, pv->yrand);

            if (facet->image.data) {
                cv::Mat to(sz, facet->image.type(), cv::Scalar(0,0,0,0));
                cv::copyMakeBorder(facet->image(cv::Rect(from_x, from_y, from_width, from_height)), to, to_y, to.rows-(to_y + to_height), to_x, to.cols-(to_x + to_width), border_type, border_value);
                //facet->image(cv::Rect(from_x, from_y, from_width, from_height)).copyTo(to(cv::Rect(to_x, to_y, to_width, to_height)));
                facet->image = to;
            }
            if (!facet->annotation.empty()) {
                int dx = to_x - from_x;
                int dy = to_y - from_y;
                facet->annotation.size = sz;
                facet->annotation.transform([dx, dy](vector<cv::Point2f> *f) {
                        for (auto &pt: *f) {
                            pt.x += dx;
                            pt.y += dy;
                        }
                });
            }

            return sizeof(PerturbVector);
        }

    };

    class ClipSquare: public Transform{
        int max_shift;

        static int pick_off (int Long, int Short, int shift) {
            int off = (Long - Short) / 2 + shift;
            if (off < 0) off = 0;
            if (off + Short > Long) off = Long - Short;
            return off;
        }
    public:
        struct PerturbVector {
            int shift;
        };

        ClipSquare (json const &spec)
            : max_shift(spec.value<int>("shift", 0))
        {
            CHECK(false);   // implement perturb
        }

        virtual size_t pv_size () const { return sizeof(PerturbVector); }

        virtual size_t pv_sample (random_engine &rng, bool perturb, void *buf) const {
            // TODO
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->shift = rng() % (max_shift * 2 + 1) - max_shift;
            return sizeof(PerturbVector);
        }

        virtual size_t apply_one (Facet *facet, bool perturb, void const *buf) const {
            // TODO
            cv::Size sz = facet->check_size();
            if (sz.width == 0) return sizeof(PerturbVector);
            if (sz.height == 0) return sizeof(PerturbVector);
            if (facet->type == Facet::FEATURE) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            int shift = pv->shift;

            int row_off = 0, col_off = 0, length = 0;
            if (sz.width < sz.height) {
                row_off = pick_off(sz.height, sz.width, shift);
                length = sz.width;
            }
            else {
                col_off = pick_off(sz.width, sz.height, shift);
                length = sz.height;
            }

            if (facet->image.data) {
                cv::Mat to(cv::Size(length, length), facet->image.type(), cv::Scalar(0,0,0,0));
                facet->image(cv::Rect(col_off, row_off, length, length)).copyTo(to);
                //facet->image(cv::Rect(from_x, from_y, from_width, from_height)).copyTo(to(cv::Rect(to_x, to_y, to_width, to_height)));
                facet->image = to;
            }
            if (!facet->annotation.empty()) {
                facet->annotation.size = cv::Size(length, length);
                facet->annotation.transform([col_off, row_off](vector<cv::Point2f> *f) {
                        for (auto &pt: *f) {
                            pt.x -= col_off;
                            pt.y -= row_off;
                        }
                });
            }
            return sizeof(PerturbVector);
        }

    };

    class RandomSquareClip: public Transform {
        // for clipping from very large images
        // for autoencoder training
        int size;
        float pad;
        int center; // disable perturb
    public:
        struct PerturbVector {
            int shift_x;
            int shift_y;
        };

        RandomSquareClip (json const &spec)
            : size(spec.value<int>("size", 512)),
            pad(spec.value<float>("pad", 0.0)),
            center(spec.value<int>("center", 0))
        {
        }

        virtual size_t pv_size () const { return sizeof(PerturbVector); }

        virtual size_t pv_sample (random_engine &rng, bool perturb, void *buf) const {
            // TODO
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->shift_x = rng();
            pv->shift_y = rng();
            return sizeof(PerturbVector);
        }

        virtual size_t apply_one (Facet *facet, bool perturb, void const *buf) const {
            // TODO
            cv::Size sz = facet->check_size();
            if (sz.width == 0) return sizeof(PerturbVector);
            if (sz.height == 0) return sizeof(PerturbVector);
            if (facet->type == Facet::FEATURE) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            if (center != 0) {
                perturb = false;
            }

            int off_x = 0, off_y = 0;
            int pad_x = 0, pad_y = 0;
            if (perturb) {
                // off_x: 0 - (sz.width - size)
                CHECK(pv->shift_x >= 0);
                CHECK(pv->shift_y >= 0);
                // sz.width - max_off_x = size
                // max_off_x = sz.width - size
                // off_x <= max_off_x
                if (sz.width >= size) {
                    off_x = pv->shift_x % (sz.width - size + 1);
                }
                else {
                    // pad_x + sz.width <= size
                    // pad_x <= size - sz.width
                    pad_x = pv->shift_x % (size - sz.width + 1);
                }
                if (sz.height >= size) {
                    off_y = pv->shift_y % (sz.height - size + 1);
                }
                else {
                    pad_y = pv->shift_y % (size - sz.height + 1);
                }
            }
            else {
                if (sz.width >= size) {
                    off_x = (sz.width - size) / 2;
                }
                else {
                    pad_x = (size - sz.width) / 2;
                }
                if (sz.height >= size) {
                    off_y = (sz.height - size) / 2;
                }
                else {
                    pad_y = (size - sz.height) / 2;
                }
            }

            if (facet->image.data) {
                cv::Mat to(cv::Size(size, size), facet->image.type(), cv::Scalar(pad,pad,pad,0));
                int copy_width = std::min<int>(size, sz.width);
                int copy_height = std::min<int>(size, sz.height);
                facet->image(cv::Rect(off_x, off_y, copy_width, copy_height)).copyTo(to(cv::Rect(pad_x, pad_y, copy_width, copy_height)));
                //facet->image(cv::Rect(from_x, from_y, from_width, from_height)).copyTo(to(cv::Rect(to_x, to_y, to_width, to_height)));
                facet->image = to;
            }
            if (!facet->annotation.empty()) {
                facet->annotation.size = cv::Size(size, size);
                facet->annotation.transform([off_x, off_y, pad_x, pad_y](vector<cv::Point2f> *f) {
                        for (auto &pt: *f) {
                            pt.x += pad_x - off_x;
                            pt.y += pad_y - off_y;
                        }
                });
            }
            return sizeof(PerturbVector);
        }
    };

    class ColorSpace: public Transform {
        int code;
        float mul0;
        float mul1;

    public:
        ColorSpace (json const &spec): code(0) {
            mul0 = spec.value<float>("mul0", 1.0);
            mul1 = spec.value<float>("mul1", 1.0);
            
            string v = spec.value<string>("code", string(""));
            if (v == "BGR2GRAY") {
                code = cv::COLOR_BGR2GRAY;
            }
            else if (v == "GRAY2BGR") {
                code = cv::COLOR_GRAY2BGR;
            }
            else if (v == "BGR2HSV") {
                code = cv::COLOR_BGR2HSV;
            }
            else if (v == "HSV2BGR") {
                code = cv::COLOR_HSV2BGR;
            }
            else if (v == "BGR2Lab") {
                code = cv::COLOR_BGR2Lab;
            }
            else if (v == "Lab2BGR") {
                code = cv::COLOR_Lab2BGR;
            }
            else if (v == "BGR2RGB") {
                code = cv::COLOR_BGR2RGB;
            }
            else CHECK(false); // << code << " not supported.";
        }
        virtual size_t apply_one (Facet *facet, bool, void const *buf) const {
            if (facet->type == Facet::IMAGE && facet->image.data) {
                if (mul0 != 1) {
                    facet->image *= mul0;
                }
                cv::Mat tmp;
                cv::cvtColor(facet->image, tmp, code);
                facet->image = tmp;
                if (mul1 != 1) {
                    facet->image *= mul1;
                }
            }
            return 0;
        }
    };

    class Sometimes: public Transform {
        // if not perturb means never
        float chance;
        Transforms transforms;
        std::uniform_real_distribution<float> linear01;
        struct PerturbVector {
            bool apply;
        };
    public:
        Sometimes (json const &spec)
            : chance(spec.value<float>("chance", 0.5)),
            transforms(spec.at("transforms")),
            linear01(0.0, 1.0) {
        }
        virtual size_t pv_size () const {
            return sizeof(PerturbVector) + transforms.pv_size(); 
        }
        virtual size_t pv_sample (random_engine &rng, bool perturb, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            float p = const_cast<Sometimes *>(this)->linear01(rng);
            pv->apply = (p <= chance) && perturb;
            if (pv->apply) {
                transforms.pv_sample(rng, perturb, static_cast<char*>(buf) + sizeof(PerturbVector));
            }
            return pv_size();
        }
        virtual size_t apply_one (Facet *facet, bool perturb, void const *buf) const {
            if (facet->type == Facet::FEATURE) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            if (pv->apply) {
                transforms.apply_one(facet, perturb, static_cast<char const*>(buf) + sizeof(PerturbVector));
            }
            return pv_size();
        }
    };

    class SomeOf: public Transform {
        // if not perturb means none
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

        virtual size_t pv_sample (random_engine &rng, bool perturb, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            std::fill(pv->apply.begin(), pv->apply.end(), false);
            if (perturb) {
                std::vector<unsigned> index(transforms.sub.size());
                for (unsigned i = 0; i < index.size(); ++i) {
                    index[i] = i;
                }
                std::shuffle(index.begin(), index.end(), rng);
                for (unsigned i = 0; i < count; ++i) {
                    pv->apply[index[i]] = true;
                }
                char *p = static_cast<char*>(buf) + sizeof(PerturbVector);
                for (unsigned i = 0; index.size(); ++i) {
                    if (pv->apply[i]) {
                        p += transforms.sub[i]->pv_sample(rng, perturb, p);
                    }
                }
            }
            return pv_size();
        }

        virtual size_t apply_one (Facet *facet, bool perturb, void const *buf) const {
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            char const *p = static_cast<char const *>(buf) + sizeof(PerturbVector);
            for (unsigned i = 0; i < transforms.sub.size(); ++i) {
                if (pv->apply[i]) {
                    p += transforms.sub[i]->apply_one(facet, perturb, p);
                }
            }
            return pv_size();
        }
    };

    class AugFlip: public Transform {
        bool horizontal;
        bool vertical;
        bool transpose;
        struct PerturbVector {
            bool horizontal;
            bool vertical;
            bool transpose;
        };
    public:
        AugFlip (json const &spec) {
            horizontal = spec.value<bool>("horizontal", false);
            vertical = spec.value<bool>("vertical", false);
            transpose = spec.value<bool>("transpose", false);
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, bool perturb, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->horizontal = horizontal && (rng() % 2) && perturb;
            pv->vertical = vertical && (rng() % 2) && perturb;
            pv->transpose = transpose && (rng() % 2) && perturb;
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, bool perturb, void const *buf) const {
            if (facet->type == Facet::FEATURE || !perturb) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            bool hflip = pv->horizontal;
            bool vflip = pv->vertical;
            bool transpose = pv->transpose;
            if (facet->image.data) {
                if (hflip && vflip) {
                    cv::flip(facet->image, facet->image, -1);
                }
                else if (hflip && !vflip) {
                    cv::flip(facet->image, facet->image, 1);
                }
                else if (!hflip && vflip) {
                    cv::flip(facet->image, facet->image, 0);
                }

                if (transpose) {
                    cv::transpose(facet->image, facet->image);
                }
            }
            if (!facet->annotation.empty()) {
                cv::Size sz = facet->annotation.size;
                if (hflip && vflip) {
                    facet->annotation.transform([sz](vector<cv::Point2f> *f) {
                        for (auto &pt: *f) {
                            pt.x = sz.width - pt.x;
                            pt.y = sz.height - pt.y;
                        }
                    });
                }
                else if (hflip && !vflip) {
                    facet->annotation.transform([sz](vector<cv::Point2f> *f) {
                        for (auto &pt: *f) {
                            pt.x = sz.width - pt.x;
                        }
                    });
                }
                else if (!hflip && vflip) {
                    facet->annotation.transform([sz](vector<cv::Point2f> *f) {
                        for (auto &pt: *f) {
                            pt.y = sz.height - pt.y;
                        }
                    });
                }

                if (transpose) {
                    int w = facet->annotation.size.width;
                    int h = facet->annotation.size.height;
                    facet->annotation.size.width = h;
                    facet->annotation.size.height = w;

                    facet->annotation.transform([sz](vector<cv::Point2f> *f) {
                        for (auto &pt: *f) {
                            float t = pt.x;
                            pt.x = pt.y;
                            pt.y = t;
                        }
                    });
                }
            }
            return sizeof(PerturbVector);
        }
    };

    class AugFlipColor: public Transform {
        struct PerturbVector {
            bool flip;
        };
    public:
        AugFlipColor (json const &spec) {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, bool perturb, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->flip = bool(rng() % 2) && perturb;
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, bool perturb, void const *buf) const {
            if (facet->type == Facet::FEATURE || !perturb) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            bool flip = pv->flip;
            if (facet->type == Facet::IMAGE && facet->image.data) {
                if (flip) {
                    //cv::flip(facet->image, facet->image, -1);
                    facet->image *= -1;
                    facet->image += 255;
                }
            }
            return sizeof(PerturbVector);
        }
    };

    class AugGaussianBlur: public Transform, public BorderConfig {
        float min, max;
        std::uniform_real_distribution<float> linear_sigma;
        struct PerturbVector {
            float sigma;
        };
    public:
        AugGaussianBlur (json const &spec)
            : BorderConfig(spec, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0)),
            linear_sigma(spec.value<float>("min", 0), spec.value<float>("max", 3)) {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, bool perturb, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            if (perturb) {
                pv->sigma = const_cast<AugGaussianBlur *>(this)->linear_sigma(rng);
            }
            else {
                pv->sigma = 0;
            }
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, bool perturb, void const *buf) const {
            if (facet->type == Facet::FEATURE || !perturb) return sizeof(PerturbVector);
            if (!(facet->type == Facet::IMAGE && facet->image.data)) return sizeof(PerturbVector);

            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            if (perturb) {
                float sigma = pv->sigma;
                int kernel = int(std::round(sigma * 3));
                if (kernel % 2 == 0) kernel += 1;
                cv::GaussianBlur(facet->image, facet->image, cv::Size(kernel, kernel), sigma, sigma, border_type);
            }
            return sizeof(PerturbVector);
        }
    };

    class AugScale: public Transform {
        float range;
        float min, max;
        std::uniform_real_distribution<float> linear_scale;
        struct PerturbVector {
            float scale;
        };
    public:
        AugScale (json const &spec):
            range(spec.value<float>("range", 0)),
            linear_scale(spec.value<float>("min", 1.0-range), spec.value<float>("max", 1.0+range)) {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, bool perturb, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            if (perturb) {
                pv->scale = const_cast<AugScale *>(this)->linear_scale(rng);
            }
            else {
                pv->scale = 1.0;
            }
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, bool perturb, void const *buf) const {
            if (facet->type == Facet::FEATURE || !perturb) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            cv::Size sz0 = facet->check_size();
            if (sz0.width == 0) return sizeof(PerturbVector);
            cv::Size sz = sz0;
            sz.width = std::round(sz.width * pv->scale);
            sz.height = std::round(sz.height * pv->scale);
            if (facet->image.data) {
                cv::resize(facet->image, facet->image, sz);
            }
            if (!facet->annotation.empty()) {
                float fx = 1.0 * sz.width / sz0.width;
                float fy = 1.0 * sz.height / sz0.height;

                facet->annotation.size = sz;

                facet->annotation.transform([fx, fy](vector<cv::Point2f> *f) {
                        for (auto &pt: *f) {
                            pt.x *= fx;
                            pt.y *= fy;
                        }
                });
            }
            return sizeof(PerturbVector);
        }
    };

    class AugAdd: public Transform {
        float range, range1, range2, range3;
        std::uniform_real_distribution<float> linear_delta1;
        std::uniform_real_distribution<float> linear_delta2;
        std::uniform_real_distribution<float> linear_delta3;
    public:
        struct PerturbVector {
            cv::Scalar delta;
        };
        AugAdd (json const &spec) :
            range(spec.value<float>("range", 0)),
            range1(spec.value<float>("range1", float(range))),
            range2(spec.value<float>("range2", float(range))),
            range3(spec.value<float>("range3", float(range))),
            linear_delta1(spec.value<float>("min1", -range1), spec.value<float>("max1", float(range1))),
            linear_delta2(spec.value<float>("min2", -range2), spec.value<float>("max2", float(range2))),
            linear_delta3(spec.value<float>("min3", -range3), spec.value<float>("max3", float(range3)))
        {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, bool perturb, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            if (perturb) {
                pv->delta = cv::Scalar(const_cast<AugAdd *>(this)->linear_delta1(rng),
                                   const_cast<AugAdd *>(this)->linear_delta2(rng),
                                   const_cast<AugAdd *>(this)->linear_delta3(rng));
            }
            else {
                pv->delta = cv::Scalar(0,0,0);
            }
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, bool perturb, void const *buf) const {
            if (facet->type == Facet::FEATURE || !perturb) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            if (facet->type == Facet::IMAGE && facet->image.data) {
                //std::cerr << pv->delta[0] << ' ' << pv->delta[1] << ' ' << pv->delta[2] << std::endl;
                facet->image += pv->delta;
            }
            return sizeof(PerturbVector);
        }
    };

    class AugRotate: public Transform, public BorderConfig {
        float range;
        std::uniform_real_distribution<float> linear_angle;
        struct PerturbVector {
            float angle;
        };
    public:
        AugRotate (json const &spec)
            : BorderConfig(spec, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0)),
            range(spec.value<float>("range", 0)),
            linear_angle(spec.value<float>("min", -range), spec.value<float>("max", 0+range)) {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, bool perturb, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            if (perturb) {
                pv->angle = const_cast<AugRotate *>(this)->linear_angle(rng);
            }
            else {
                pv->angle = 0;
            }
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, bool perturb, void const *buf) const {
            if (facet->type == Facet::FEATURE || !perturb) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            cv::Size sz = facet->check_size();
            if (sz.width == 0) return sizeof(PerturbVector);
            cv::Mat rot = cv::getRotationMatrix2D(cv::Point(sz.width/2, sz.height/2), pv->angle, 1.0);
            if (facet->image.data) {
                cv::Mat tmp;
                cv::warpAffine(facet->image, tmp, rot, sz, facet->type == Facet::LABEL ? cv::INTER_NEAREST: cv::INTER_LINEAR, border_type, border_value);//, cv::INTER);
                //cv::resize(tmp, tmp, cv::Size(), p.scale, p.scale);
                facet->image = tmp;
            }
            if (!facet->annotation.empty()) {
                facet->annotation.transform([&rot](vector<cv::Point2f> *f) {
                        cv::transform(*f, *f, rot);
                });
            }

            return sizeof(PerturbVector);
        }
    };

    class AugAffine: public Transform {
    public:
        AugAffine (json const &spec) {
            CHECK(false);
        }
    };

    struct CircleAnchor {

        static unsigned constexpr PRIOR_PARAMS = 1;
        static unsigned constexpr PARAMS = 3;

        struct Shape {
            cv::Point2f center;
            float radius;
            float weight;
        };

        static void init_prior (float *params) {
            params[0] = 0;
        }

        static void init_prior(float *params, json const &p) {
            CHECK(0);// << "Not supported.";
        }


        static void init_shape_with_controls (Shape *c, vector<cv::Point2f> const &ctrls, bool weighted) {
            float minx = ctrls[0].x;
            float maxx = minx;
            float miny = ctrls[0].y;
            float maxy = miny;
            for (unsigned j = 1; j < ctrls.size(); ++j) {
                minx = std::min(ctrls[j].x, minx);
                maxx = std::max(ctrls[j].x, maxx);
                miny = std::min(ctrls[j].y, miny);
                maxy = std::max(ctrls[j].y, maxy);
            }
            cv::Point2f ul(minx, miny);
            cv::Point2f br(maxx, maxy);
            c->center = (ul + br);
            c->center.x /=  2; 
            c->center.y /=  2;
            c->radius =  std::sqrt((maxx-minx)*(maxy-miny)) / 2;
            c->weight = 1.0;
            if (weighted) {
                c->weight = 1.0 / (c->radius * c->radius);
            }
        }

        static float score (cv::Point_<float> const &pt,
                              float const *prior,
                              Shape const &c) {
            float r = cv::norm(pt - c.center);
            return (c.radius - r) / c.radius;
        }

        static void update_params (cv::Point_<float> const &pt, Shape const &c, float *params) {
            params[0] = c.center.x - pt.x;
            params[1] = c.center.y - pt.y;
            params[2] = c.radius;
        }
    };

    struct BoxAnchor {
        static unsigned constexpr PRIOR_PARAMS = 2;
        // params: width height
        static unsigned constexpr PARAMS = 4;

        struct Shape {
            cv::Point2f center;
            float width, height;
            float weight;
        };

        static void init_prior (float *params) {
            params[0] = params[1] = 0;
        }

        static void init_prior(float *params, json const &p) {
            CHECK(p.is_array());
            CHECK(p.size() == 2);
            float w = p.at(0);
            float h = p.at(1);
            params[0] = w;
            params[1] = h;
            logging::warn("prior w: {} h: {}", params[0], params[1]);
        }

        static void init_shape_with_controls (Shape *c, vector<cv::Point2f> const &ctrls, bool weighted) {
            float minx = ctrls[0].x;
            float maxx = minx;
            float miny = ctrls[0].y;
            float maxy = miny;
            for (unsigned j = 1; j < ctrls.size(); ++j) {
                minx = std::min(ctrls[j].x, minx);
                maxx = std::max(ctrls[j].x, maxx);
                miny = std::min(ctrls[j].y, miny);
                maxy = std::max(ctrls[j].y, maxy);
            }
            cv::Point2f ul(minx, miny);
            cv::Point2f br(maxx, maxy);
            c->center = (ul + br);
            c->center.x /=  2; 
            c->center.y /=  2;
            c->width = maxx - minx;
            c->height = maxy - miny;
            c->weight = 1.0;
            if (weighted) {
                c->weight = 1.0 / (c->width * c->height);
            }
            CHECK(c->weight > 0);
        }

        static cv::Rect_<float> make_rect (cv::Point2f const &pt,
                                    float width, float height) {
            return cv::Rect_<float>(pt.x - width/2,
                                    pt.y - height/2,
                                    width, height);
        }

        static float score (cv::Point2f const &pt,
                              float const *prior,
                              Shape const &c) {
            if (prior[0] == 0) {
                float dx = std::abs(pt.x - c.center.x) * 2; 
                float dy = std::abs(pt.y - c.center.y) * 2;
                return std::min((c.width - dx) / c.width,
                                        (c.height - dy) / c.height);
            }
            // pt, prior[0], prior[1]
            // c
            cv::Rect_<float> p = make_rect(pt, prior[0], prior[1]);
            cv::Rect_<float> r = make_rect(c.center, c.width, c.height);

            cv::Rect_<float> u = p & r;
            float i = u.area();
            return i / (p.area() + r.area() - i + 0.00001);
        }

        static void update_params (cv::Point_<float> const &pt, Shape const &c, float *params) {
            params[0] = c.center.x - pt.x;
            params[1] = c.center.y - pt.y;
            params[2] = c.width;
            params[3] = c.height;
            CHECK(c.width >= 1);
            CHECK(c.height >= 1);
        }
    };

    struct PointAnchor {
        static unsigned constexpr PRIOR_PARAMS = 1; // not used
        // params: width height
        static unsigned constexpr PARAMS = 2;

        struct Shape: public cv::Point2f {
            float weight;
        };

        static void init_prior (float *params) {
            params[0] = 0;
        }

        static void init_prior(float *params, json const &p) {
            // not supported
            CHECK(false);
        }

        static void init_shape_with_controls (Shape *c, vector<cv::Point2f> const &ctrls, bool) {
            CHECK(ctrls.size() == 1);
            c->x = ctrls[0].x;
            c->y = ctrls[0].y;
            c->weight = 1.0;
        }

        static float score (cv::Point2f const &pt,
                              float const *prior,
                              Shape const &c) {
            float dx = pt.x - c.x;
            float dy = pt.y - c.y;
            return std::exp(-std::sqrt(dx*dx+dy*dy));
        }

        static void update_params (cv::Point_<float> const &pt, Shape const &c, float *params) {
            params[0] = c.x - pt.x;
            params[1] = c.y - pt.y;
        }
    };

    template <typename ANCHOR>
    class DenseAnchors: public Transform {
        int index;		// annotation facet
        int downsize;
        bool weighted;
        cv::Mat priors;
        float upper_th;
        float lower_th;
        float params_default;

        struct Shape: public ANCHOR::Shape {
            // keep track the best match of the shape
            // this is always used if score > lower_th
            float score;
            float const *prior;
            cv::Scalar color;
            float tag;
            cv::Point_<float> pt;
            float *label;
            float *label_mask;
            float *params;
            float *params_mask;
            Shape (): score(0), prior(nullptr),
                label(nullptr), label_mask(nullptr), 
                params(nullptr), params_mask(nullptr) {
            }
        };

    public:
        DenseAnchors (json const &spec) { //: priors(1, ANCHOR::PRIOR_PARAMS, CV_32F, cv::Scalar(0)) {
            index = spec.value<int>("index", 1);
            downsize = spec.value<int>("downsize", 1);
            upper_th = spec.value<float>("upper_th", 0.6);
            lower_th = spec.value<float>("lower_th", 0.4);
            weighted = spec.value<bool>("weighted", false);
            params_default = spec.value<float>("params_default", 0);
            // priors is an array
            // each entry will construct an anchor object
            //
            if (spec.find("priors") != spec.end()) {
                auto const &pspec = spec.at("priors");
                if (pspec.size() > 0) {
                    priors.create(pspec.size(), ANCHOR::PRIOR_PARAMS, CV_32F);
                    unsigned i = 0;
                    for (auto const &p: spec.at("priors")) {
                        ANCHOR::init_prior(priors.ptr<float>(i), p);

                        ++i;
                    }
                }
            }
            if (priors.empty()) {
                priors.create(1, ANCHOR::PRIOR_PARAMS, CV_32F);
                ANCHOR::init_prior(priors.ptr<float>(0));
            }
        }

        virtual size_t apply (Sample *sample, bool, void const *) const {
            auto const &facet = sample->facets[index];
            auto const &anno = facet.annotation;

            CHECK(!anno.empty());

            vector<Shape> truths(anno.shapes.size());  // ground-truth circles
            for (unsigned i = 0; i < anno.shapes.size(); ++i) {
                vector<cv::Point2f> const &ctrls = anno.shapes[i]->__controls();
                CHECK(ctrls.size() >= 1); // must be boxes
                truths[i].color = anno.shapes[i]->color;
                truths[i].tag = anno.shapes[i]->tag;
                ANCHOR::init_shape_with_controls(&truths[i], ctrls, weighted);
            }

            // params: dx dy radius
            cv::Size sz = anno.size;
            CHECK(sz.width % downsize == 0);
            CHECK(sz.height % downsize == 0);
            sz.width /= downsize;
            sz.height /= downsize;

            cv::Mat label(sz, CV_32FC(priors.rows), cv::Scalar(0));
            // only effective for near and far points
            cv::Mat label_mask(sz, CV_32FC(priors.rows));
            // only effective for near points
            cv::Mat params(cv::Mat::zeros(sz, CV_32FC(priors.rows * ANCHOR::PARAMS)));
            cv::Mat params_mask(cv::Mat::zeros(sz, CV_32FC(priors.rows)));

            {
                float *b = label_mask.ptr<float>(0);
                float *e = b + label_mask.total() * label_mask.channels();
                CHECK(e - b == label_mask.rows * label_mask.cols * priors.rows);
                std::fill(b, e, 1.0);
            }

            if (params_default != 0) {
                float *b = params.ptr<float>(0);
                float *e = b + params.total() * params.channels();
                CHECK(params.total() == int(sz.area()));
                CHECK(params.channels() == int(priors.rows * ANCHOR::PARAMS));
                std::fill(b, e, params_default);
            }

            for (int y = 0; y < sz.height; ++y) {

                float *pl = label.ptr<float>(y);
                float *plm = label_mask.ptr<float>(y);
                float *pp = params.ptr<float>(y);
                float *ppm = params_mask.ptr<float>(y);

                for (int x = 0; x < sz.width; ++x) {
                    // find closest shape
                    cv::Point2f pt(x * downsize, y * downsize);

                    for (int k = 0; k < priors.rows; ++k,
                            pl += 1, plm +=1,
                            pp += ANCHOR::PARAMS, ppm += 1) {
                        float const *prior = priors.ptr<float>(k);

                        Shape *best_c = nullptr;
                        float best_d = 0;
                        for (auto &c: truths) {
                            // TODO: what if a pixel belongs to two shapes
                            float d = ANCHOR::score(pt, prior, c); 
                            if (d > best_d) {   // find best circle
                                best_d = d;
                                best_c = &c;
                            }
                            if (d > c.score) {
                                c.score = d;
                                c.pt = pt;
                                c.prior = prior;
                                c.label = pl;
                                c.label_mask = plm;
                                c.params = pp;
                                c.params_mask = ppm;
                            }
                        }
                        if (!best_c) continue;
                        if (best_d >= lower_th) {
                            ANCHOR::update_params(pt, *best_c, pp);
                            ppm[0] = best_c->weight;
                            if (best_d < upper_th) {
                                plm[0] = 0;
                            }
                            else {
                                pl[0] = 1;      // to class label
                            }
                        }
                    } // prior
                } // x
            } // y
#if 0
            for (auto &c: truths) {
                if (c.label == nullptr) {
                    //LOG(DEBUG)<< "MISS: ";// << c.center.x << ',' << c.center.y << ' ' << c.radius << std::endl;
                    continue;   // not found
                }
                if (c.score >= lower_th) {
                    c.label[0] = 1;
                    c.label_mask[0] = 1;
                    ANCHOR::update_params(c.pt, c, c.params);
                    c.params_mask[0] = c.weight;
                }
            }
#endif
            sample->facets.emplace_back(label, Facet::LABEL);
            sample->facets.emplace_back(label_mask, Facet::LABEL);
            sample->facets.emplace_back(params, Facet::LABEL);
            sample->facets.emplace_back(params_mask, Facet::LABEL);
            return 0;
        }
    };

    class BoxFeature: public Transform {
        int index;
        float min_area;
    public:
        BoxFeature (json const &spec) {
            index = spec.value<int>("index", 1);
            min_area = spec.value<float>("min_area", 1);
        }

        virtual size_t apply (Sample *sample, bool, void const *) const {
            auto const &facet = sample->facets[index];
            auto const &anno = facet.annotation;

            CHECK(!anno.empty());

            cv::Mat feature(anno.shapes.size(), 6, CV_32F);
            unsigned o = 0;
            for (unsigned i = 0; i < anno.shapes.size(); ++i) {
                vector<cv::Point2f> const &ctrls = anno.shapes[i]->__controls();
                CHECK(ctrls.size() >= 1); // must be boxes
                BoxAnchor::Shape box;
                BoxAnchor::init_shape_with_controls(&box, ctrls, false);
                float *p = feature.ptr<float>(o);
                p[0] = anno.shapes[i]->color[0];
                p[1] = anno.shapes[i]->tag;
                p[2] = box.center.x - box.width/2;
                p[3] = box.center.y - box.height/2;
                p[4] = p[2] + box.width;
                p[5] = p[3] + box.height;

                p[2] = std::min<float>(anno.size.width-1, std::max<float>(0, p[2]));
                p[3] = std::min<float>(anno.size.height-1, std::max<float>(0, p[3]));
                p[4] = std::min<float>(anno.size.width-1, std::max<float>(0, p[4]));
                p[5] = std::min<float>(anno.size.height-1, std::max<float>(0, p[5]));
                float area = (p[4] - p[2]  + 1) * (p[5] - p[3] + 1);
                if (area >= min_area) {
                     o += 1;
                }
            }
            if (o < anno.shapes.size()) {
                feature = feature(cv::Rect(0, 0, 6, o));
            }
            sample->facets.emplace_back(feature , Facet::FEATURE);
            return 0;
        }
    };

    class BasicKeyPoints: public Transform {
        int index;
        int classes;    // output channels
        int stride;
        float radius;

        static float dist (float x, float y, int cx, int cy) {
            x -= cx;
            x *= x;
            y -= cy;
            y *= y;
            return std::sqrt(x + y);
        }
    public:
        BasicKeyPoints (json const &spec) {
            index = spec.value<int>("index", 1);
            classes = spec.value<int>("classes", 1);
            stride = spec.value<int>("downsize", 1);
            radius = spec.value<float>("radius", 5);
            CHECK(classes * 2 <= CV_CN_MAX); // 512
        }

        virtual size_t apply (Sample *sample, bool, void const *) const {
            auto const &facet = sample->facets[index];
            auto const &anno = facet.annotation;

            CHECK(!anno.empty());
            //cv::Mat label(sz, CV_32FC(priors.rows), cv::Scalar(0));
            cv::Size sz = anno.size;
            CHECK(sz.width % stride == 0);
            CHECK(sz.height % stride == 0);
            sz.width /= stride;
            sz.height /= stride;
            cv::Mat mask(sz, CV_32FC(classes), CV_32F);         // 0/1
            cv::Mat offset(sz, CV_32FC(classes*2), CV_32F);     // x,y
            // initialize to zero
            // don't know how to handle multiple-channel element in opencv
            CHECK(mask.isContinuous());
            bzero(mask.ptr<uint8_t>(0), mask.total() * mask.elemSize()); 
            CHECK(offset.isContinuous());
            bzero(offset.ptr<uint8_t>(0), offset.total() * offset.elemSize()); 

            for (unsigned i = 0; i < anno.shapes.size(); ++i) {
                vector<cv::Point2f> const &ctrls = anno.shapes[i]->__controls();
                if (ctrls.empty()) continue;
                int cls = int(anno.shapes[i]->color[0])-1;
                for (cv::Point2f pt: ctrls) {
                    // draw circle
                    int x0 = std::max(0, int(floorf(pt.x / stride - radius)));
                    int x1 = std::min(sz.width-1, int(ceilf(pt.x / stride + radius)));

                    int y0 = std::max(0, int(floorf(pt.y / stride - radius)));
                    int y1 = std::min(sz.height-1, int(ceilf(pt.y / stride + radius)));

                    for (int y = y0; y <= y1; ++y) {
                        float *pm = mask.ptr<float>(y) + x0 * classes + cls;
                        float *po = offset.ptr<float>(y) + x0 * classes * 2 + cls * 2;
                        int ys = y * stride;
                        for (int x = x0; x <= x1; ++x, pm += classes, po += classes * 2) {
                            int xs = x * stride;
                            if (dist(pt.x, pt.y, xs, ys) <= radius) {
                                pm[0] = 1.0;
                                po[0] = pt.x - xs;
                                po[1] = pt.y - ys;
                            }
                        }
                    }
                }
            }
            sample->facets.emplace_back(mask, Facet::LABEL);
            sample->facets.emplace_back(offset, Facet::LABEL);
            return 0;
        }
    };

#if 0
    <template typename=ANCHOR>
    class DrawDenseAnchors: public Transform {
        int index;
        int upsize;
        int copy;
    public:
        DrawDenseCircleAnchors (json const &spec) {
            index = spec.value<int>("index", 4);
            upsize = spec.value<int>("upsize", 1);
            copy = spec.value<int>("copy", 0);
        }
        virtual size_t apply (Sample *sample, void const *) const {
            auto const &params = sample->facets[index].image;
            auto const &params_mask = sample->facets[index+1].image;

            cv::Size sz = params.size();
            sz.width *= upsize;
            sz.height *= upsize;

            cv::Mat image = sample->facets[0].image.clone();
            CHECK(image.channels() == 3);

            for (int y = 0; y < params.rows; ++y) {
                float const *pp = params.ptr<float>(y);
                float const *pm = params_mask.ptr<float>(y);
                for (int x = 0; x < params.cols; ++x, pp += 3, pm += 1) {
                    if (pm[0] == 0) continue;
                    cv::Point2f pt = cv::Point2f(x, y) + cv::Point2f(pp[1],pp[2]);
                    pt *= upsize;
                    cv::circle(image, round(pt), std::round(pp[0] * upsize), cv::Scalar(0, 255, 0), 1);

                }
            }
            sample->facets.emplace_back(image);
            return 0;
        }
    };
#endif

    // apply to all channels
#if 0
    class WaveAugAdd: public Transform {
        float range;
        std::uniform_real_distribution<float> linear_delta;
    public:
        struct PerturbVector {
            float delta;
        };
        WaveAugAdd (json const &spec) :
            range(spec.value<float>("range", 0)),
            linear_delta(spec.value<float>("min", -range), spec.value<float>("max", 0+range))
        {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->delta = const_cast<WaveAugAdd *>(this)->linear_delta(rng);
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, void const *buf) const {
            if (facet->type == Facet::FEATURE) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            if (facet->type == Facet::IMAGE && facet->image.data) {
                //std::cerr << pv->delta[0] << ' ' << pv->delta[1] << ' ' << pv->delta[2] << std::endl;
                facet->image += pv->delta;
            }
            return sizeof(PerturbVector);
        }
    };

    class WaveAugMul: public Transform {
        float range;
        std::uniform_real_distribution<float> linear_delta;
    public:
        struct PerturbVector {
            float delta;
        };
        WaveAugMul (json const &spec) :
            range(spec.value<float>("range", 0)),
            linear_delta(spec.value<float>("min", 1.0-range), spec.value<float>("max", 1.0+range))
        {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->delta = const_cast<WaveAugMul *>(this)->linear_delta(rng);
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, void const *buf) const {
            if (facet->type == Facet::FEATURE) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            if (facet->type == Facet::IMAGE && facet->image.data) {
                //std::cerr << pv->delta[0] << ' ' << pv->delta[1] << ' ' << pv->delta[2] << std::endl;
                facet->image *= pv->delta;
            }
            return sizeof(PerturbVector);
        }
    };

    class AugWidthScale: public AugScale {
        float range;
        std::uniform_real_distribution<float> linear_scale;
        struct PerturbVector {
            float scale;
        };
    public:
        AugWidthScale (json const &spec): AugScale(spec) {
        }
        virtual size_t apply_one (Facet *facet, void const *buf) const {
            if (facet->type == Facet::FEATURE) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            cv::Size sz0 = facet->check_size();
            CHECK(sz0.height == 1);
            if (sz0.width == 0) return sizeof(PerturbVector);
            cv::Size sz = sz0;
            sz.width = std::round(sz.width * pv->scale);
            if (facet->image.data) {
                cv::resize(facet->image, facet->image, sz);
            }
            if (!facet->annotation.empty()) {
                CHECK(0);
            }
            return sizeof(PerturbVector);
        }
    };
#endif

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
        if (type == "mask") {
            return std::unique_ptr<Transform>(new Mask(spec));
        }
        if (type == "erode_mask") {
            return std::unique_ptr<Transform>(new ErodeMask(spec));
        }
        if (type == "rasterize") {
            return std::unique_ptr<Transform>(new Rasterize(spec));
        }
        if (type == "normalize") {
            return std::unique_ptr<Transform>(new Normalize(spec));
        }
        else if (type == "resize") {
            return std::unique_ptr<Transform>(new Resize(spec));
        }
        else if (type == "scale") {
            return std::unique_ptr<Transform>(new Scale(spec));
        }
        else if (type == "pyramid") {
            return std::unique_ptr<Transform>(new Pyramid(spec));
        }
        else if (type == "pad") {
            return std::unique_ptr<Transform>(new Pad(spec));
        }
        else if (type == "clip") {
            return std::unique_ptr<Transform>(new Clip(spec));
        }
        else if (type == "clip_square") {
            return std::unique_ptr<Transform>(new ClipSquare(spec));
        }
        else if (type == "random_square_clip") {
            return std::unique_ptr<Transform>(new RandomSquareClip(spec));
        }
        else if (type == "colorspace") {
            return std::unique_ptr<Transform>(new ColorSpace(spec));
        }
        else if (type == "augment.flip") {
            return std::unique_ptr<Transform>(new AugFlip(spec));
        }
        else if (type == "augment.flip_color") {
            return std::unique_ptr<Transform>(new AugFlipColor(spec));
        }
        else if (type == "augment.blur") {
            return std::unique_ptr<Transform>(new AugGaussianBlur(spec));
        }
        else if (type == "augment.scale") {
            return std::unique_ptr<Transform>(new AugScale(spec));
        }
        else if (type == "augment.rotate") {
            return std::unique_ptr<Transform>(new AugRotate(spec));
        }
        else if (type == "augment.affine") {
            return std::unique_ptr<Transform>(new AugAffine(spec));
        }
        else if (type == "augment.add") {
            return std::unique_ptr<Transform>(new AugAdd(spec));
        }
        else if (type == "anchors.dense.point") {
            return std::unique_ptr<Transform>(new DenseAnchors<PointAnchor>(spec));
        }
        else if (type == "anchors.dense.circle") {
            return std::unique_ptr<Transform>(new DenseAnchors<CircleAnchor>(spec));
        }
        else if (type == "anchors.dense.box") {
            return std::unique_ptr<Transform>(new DenseAnchors<BoxAnchor>(spec));
        }
        else if (type == "keypoints.basic") {
            return std::unique_ptr<Transform>(new BasicKeyPoints(spec));
        }
        else if (type == "box_feature") {
            return std::unique_ptr<Transform>(new BoxFeature(spec));
        }
        /*
        else if (type == "anchors.dense.circle.draw") {
            return std::unique_ptr<Transform>(new DrawDenseCircleAnchors(spec));
        }
        */
#if 0
        else if (type == "wave.augment.add") {
            return std::unique_ptr<Transform>(new WaveAugAdd(spec));
        }
        else if (type == "wave.augment.mul") {
            return std::unique_ptr<Transform>(new WaveAugMul(spec));
        }
        else if (type == "wave.augment.scale") {
            return std::unique_ptr<Transform>(new AugWidthScale(spec));
        }
#endif
        else if (type == "drop") {
            return std::unique_ptr<Transform>(new Drop(spec));
        }
        else if (type == "featurize") {
            return std::unique_ptr<Transform>(new Featurize(spec));
        }
        else {
            CHECK(0);// << "unknown transformation: " << type;
        }
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
