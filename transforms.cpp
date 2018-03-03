#include "picpac-image.h"

namespace picpac {

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
            int dtype = dtype_np2cv(spec.value<string>("dtype", "uint8"));
            type = CV_MAKETYPE(dtype, channels);
            opt.use_palette = spec.value<bool>("use_palette", opt.use_palette);
            opt.thickness = spec.value<int>("thickness", opt.thickness);
        }
        virtual size_t apply (Sample *sample, void const *) const {
            auto &facet = sample->facets[index];
            if (copy >= 0) {
                facet.image = sample->facets[copy].image.clone();
            }
            else {
                facet.image = cv::Mat(facet.annotation.size, type, cv::Scalar(0,0,0,0));
                //std::cerr << facet.image.cols << 'x' << facet.image.rows << std::endl;
            }
            facet.annotation.render(&facet.image, opt);
            facet.annotation.clear();
            return 0;
        }
    };

    class Clip: public Transform {
        int min, max;
        int min_width, max_width;
        int min_height, max_height;
        int round;

        static void adjust_crop_pad_range (int &from_x, int &from_width,
                                    int &to_x, int &to_width, bool perturb, int shiftx) {
            if (from_width < to_width) {
                int margin = to_width - from_width;
                if (perturb) {
                    to_x = shiftx % margin;
                }
                else {
                    to_x = margin / 2;
                }
                to_width = from_width;
            }
            else if (from_width > to_width) {
                int margin = from_width - to_width;
                if (perturb) {
                    from_x = shiftx % margin;
                }
                else {
                    from_x = margin / 2;
                }
                from_width = to_width;
            }
        }
    public:
        struct PerturbVector {
            int xshift;
            int yshift;
        };

        Clip (json const &spec)
            : min(spec.value<int>("min", 0)),
              max(spec.value<int>("max", numeric_limits<int>::max())),
              min_width(spec.value<int>("min_width", min)),
              max_width(spec.value<int>("max_width", max)),
              min_height(spec.value<int>("min_height", min)),
              max_height(spec.value<int>("max_height", max)),
              round(spec.value<int>("round", 0))
        {
            CHECK(min_width <= max_width);
            CHECK(min_height <= max_height);
        }

        virtual size_t pv_size () const { return sizeof(PerturbVector); }

        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->xshift = rng();
            pv->yshift = rng();
            return sizeof(PerturbVector);
        }

        virtual size_t apply_one (Facet *facet, void const *buf) const {
            cv::Size sz0 = facet->check_size();
            if (sz0.width == 0) return sizeof(PerturbVector);
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            cv::Size sz = sz0;
            if (sz.width < min_width) sz.width = min_width;
            if (sz.width > max_width) sz.width = max_width;
            if (sz.height < min_height) sz.height = min_height;
            if (sz.height > max_height) sz.height = max_height;
            if (round > 0) {
                sz.width = sz.width / round * round;
                sz.height = sz.height / round * round;
            }
            if (sz == sz0) return sizeof(PerturbVector);

            int from_x = 0, from_width = sz0.width;
            int from_y = 0, from_height = sz0.height;

            int to_x = 0, to_width = sz.width;
            int to_y = 0, to_height = sz.height;

            adjust_crop_pad_range(from_x, from_width, to_x, to_width, true, pv->xshift);
            adjust_crop_pad_range(from_y, from_height, to_y, to_height, true, pv->yshift);

            if (facet->image.data) {
                cv::Mat to(sz, facet->image.type(), cv::Scalar(0,0,0,0));
                facet->image(cv::Rect(from_x, from_y, from_width, from_height)).copyTo(to(cv::Rect(to_x, to_y, to_width, to_height)));
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

    class ColorSpace: public Transform {
        int code;
        float mul0;
        float mul1;

    public:
        ColorSpace (json const &spec): code(0) {
            mul0 = spec.value<float>("mul0", 1.0);
            mul1 = spec.value<float>("mul1", 1.0);
            
            string v = spec.value<string>("code", "");
            if (v == "BGR2GRAY") {
                code = CV_BGR2GRAY;
            }
            else if (v == "GRAY2BGR") {
                code = CV_GRAY2BGR;
            }
            else if (v == "BGR2HSV") {
                code = CV_BGR2HSV;
            }
            else if (v == "HSV2BGR") {
                code = CV_HSV2BGR;
            }
            else if (v == "BGR2Lab") {
                code = CV_BGR2Lab;
            }
            else if (v == "Lab2BGR") {
                code = CV_Lab2BGR;
            }
            else if (v == "BGR2RGB") {
                code = CV_BGR2RGB;
            }
            else CHECK(false) << code << " not supported.";
        }
        virtual size_t apply_one (Facet *facet, void const *buf) const {
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


    class AugFlip: public Transform {
        bool horizontal;
        bool vertical;
        struct PerturbVector {
            bool horizontal;
            bool vertical;
        };
    public:
        AugFlip (json const &spec) {
            horizontal = spec.value<bool>("horizontal", true);
            vertical = spec.value<bool>("vertical", true);
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->horizontal = horizontal && (rng() % 2);
            pv->vertical = vertical && (rng() % 2);
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, void const *buf) const {
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            bool hflip = pv->horizontal;
            bool vflip = pv->vertical;
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
            }
            return sizeof(PerturbVector);
        }
    };

    class AugScale: public Transform {
        float range;
        std::uniform_real_distribution<float> linear_scale;
        struct PerturbVector {
            float scale;
        };
    public:
        AugScale (json const &spec):
            range(spec.value("range", 0)),
            linear_scale(spec.value<float>("min", 1.0-range), spec.value<float>("max", 1.0+range)) {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->scale = const_cast<AugScale *>(this)->linear_scale(rng);
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, void const *buf) const {
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
                float fy = 1.0 * sz.height / sz0.width;

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
            range1(spec.value<float>("range1", range)),
            range2(spec.value<float>("range2", range)),
            range3(spec.value<float>("range3", range)),
            linear_delta1(spec.value<float>("min1", -range1), spec.value<float>("max1", range1)),
            linear_delta2(spec.value<float>("min2", -range2), spec.value<float>("max2", range2)),
            linear_delta3(spec.value<float>("min3", -range3), spec.value<float>("max3", range3))
        {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->delta = cv::Scalar(const_cast<AugAdd *>(this)->linear_delta1(rng),
                                   const_cast<AugAdd *>(this)->linear_delta2(rng),
                                   const_cast<AugAdd *>(this)->linear_delta3(rng));
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, void const *buf) const {
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            if (facet->type == Facet::IMAGE && facet->image.data) {
                //std::cerr << pv->delta[0] << ' ' << pv->delta[1] << ' ' << pv->delta[2] << std::endl;
                facet->image += pv->delta;
            }
            return sizeof(PerturbVector);
        }
    };

    class AugRotate: public Transform {
        float range;
        std::uniform_real_distribution<float> linear_angle;
        struct PerturbVector {
            float angle;
        };
    public:
        AugRotate (json const &spec)
            : range(spec.value("range", 0)),
            linear_angle(spec.value("min", -range), spec.value("max", range)) {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->angle = const_cast<AugRotate *>(this)->linear_angle(rng);
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, void const *buf) const {
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            cv::Size sz = facet->check_size();
            if (sz.width == 0) return sizeof(PerturbVector);
            cv::Mat rot = cv::getRotationMatrix2D(cv::Point(sz.width/2, sz.height/2), pv->angle, 1.0);
            if (facet->image.data) {
                cv::Mat tmp;
                cv::warpAffine(facet->image, tmp, rot, sz, facet->type == Facet::LABEL ? CV_INTER_NN: CV_INTER_LINEAR);//, CV_INTER);
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

    class CircleAnchors: public Transform {
        int index;
        int downsize;
        float upper_th;
        float lower_th;

        struct Circle {
            cv::Point2f center;
            float radius;

            float score;
            uint8_t *label;
            float *label_mask;
            float *params;
            float *params_mask;
            cv::Point2f shift;
        };

    public:
        CircleAnchors (json const &spec) {
            index = spec.value<int>("index", 1);
            downsize = spec.value<int>("downsize", 1);
            upper_th = spec.value<float>("upper_th", 0.8);
            lower_th = spec.value<float>("lower_th", 0.4);
        }

        virtual size_t apply (Sample *sample, void const *) const {
            auto const &facet = sample->facets[index];
            auto const &anno = facet.annotation;

            CHECK(!anno.empty());

            vector<Circle> truths(anno.shapes.size());  // ground-truth circles
            for (unsigned i = 0; i < anno.shapes.size(); ++i) {
                vector<cv::Point2f> const &ctrls = anno.shapes[i]->__controls();
                CHECK(ctrls.size() >= 1); // must be boxes

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

                Circle &c = truths[i];

#if 0
                c.center = (ctrls[0] + ctrls[1]) / 2 / downsize;
                c.radius =  cv::norm(ctrls[0] - ctrls[1]) / 2 / downsize;
#endif
                c.center = (ul + br);
                c.center.x /=  2 * downsize;
                c.center.y /=  2 * downsize;
                c.radius =  cv::norm(br - ul) / 2 / downsize;

                c.score = numeric_limits<float>::max();
                c.label = nullptr;
                c.label_mask = nullptr;
                c.params = nullptr;
                c.params_mask = nullptr;
            }

            // params: dx dy radius
            cv::Size sz = anno.size;
            CHECK(sz.width % downsize == 0);
            CHECK(sz.height % downsize == 0);
            sz.width /= downsize;
            sz.height /= downsize;

            cv::Mat label(sz, CV_8UC1, cv::Scalar(0));
            // only effective for near and far points
            cv::Mat label_mask(sz, CV_32F, cv::Scalar(1));
            // only effective for near points
            cv::Mat params(sz, CV_32FC3, cv::Scalar(0, 0, 0));
            cv::Mat params_mask(sz, CV_32FC3, cv::Scalar(0));

            for (int y = 0; y < sz.height; ++y) {

                uint8_t *pl = label.ptr<uint8_t>(y);
                float *plm = label_mask.ptr<float>(y);
                float *pp = params.ptr<float>(y);
                float *ppm = params_mask.ptr<float>(y);

                for (int x = 0; x < sz.width; ++x, pl += 1, plm += 1, pp += 3, ppm += 3) {
                    // find closest circle
                    cv::Point2f pt(x, y);
                    Circle *best_c = nullptr;
                    float best_d = numeric_limits<float>::max();
                    for (auto &c: truths) {
                        float d = cv::norm(pt - c.center);
                        if (d < best_d) {   // find best circle
                            best_d = d;
                            best_c = &c;
                        }
                        if (d < c.score) {  // for each circle find best match
                            c.score = d;
                            c.label = pl;
                            c.label_mask = plm;
                            c.params = pp;
                            c.params_mask = ppm;
                            c.shift = c.center - pt;
                        }
                    }
                    if (!best_c) continue;
                    float r = best_d / best_c->radius;
                    if (r <= upper_th) { // close to center
                        pl[0] = 1;
                        pp[0] = best_c->radius;
                        pp[1] = best_c->center.x - x;
                        pp[2] = best_c->center.y - y;
                        ppm[0] = 1;
                        ppm[1] = 1;
                        ppm[2] = 1;
                        if (r > lower_th) {
                            plm[0] = 0;
                        }
                    }
                }
            }
            for (auto &c: truths) {
                if (c.label == nullptr) {
                    std::cerr << "MISS: " << c.center.x << ',' << c.center.y << ' ' << c.radius << std::endl;
                    continue;   // not found
                }
#if 0
                if (c.score / c.radious <= lower_th) {
                    c.label[0] = 1;
                    c.label_mask[0] = 1;
                    c.params[0] = c.radious;
                    c.params[1] = c.shift.x;
                    c.params[2] = c.shift.y;
                    c.params_mask[0] = 1;
                }
#endif
            }
            sample->facets.emplace_back(label);
            sample->facets.emplace_back(label_mask);
            sample->facets.emplace_back(params);
            sample->facets.emplace_back(params_mask);
            return 0;
        }
    };

    class DrawCircleAnchors: public Transform {
        int index;
        int upsize;
        int copy;
    public:
        DrawCircleAnchors (json const &spec) {
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

    std::unique_ptr<Transform> Transform::create (json const &spec) {
        string type = spec.at("type").get<string>();
        if (type == "rasterize") {
            return std::unique_ptr<Transform>(new Rasterize(spec));
        }
        else if (type == "clip") {
            return std::unique_ptr<Transform>(new Clip(spec));
        }
        else if (type == "colorspace") {
            return std::unique_ptr<Transform>(new ColorSpace(spec));
        }
        else if (type == "augment.flip") {
            return std::unique_ptr<Transform>(new AugFlip(spec));
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
        else if (type == "anchor.circles") {
            return std::unique_ptr<Transform>(new CircleAnchors(spec));
        }
        else if (type == "anchor.circles.draw") {
            return std::unique_ptr<Transform>(new DrawCircleAnchors(spec));
        }
        else {
            CHECK(0) << "unknown shape: " << type;
        }
        return nullptr;
    }
}
