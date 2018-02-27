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
            type = spec.value<int>("type", CV_8UC3);
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
                std::cerr << facet.image.cols << 'x' << facet.image.rows << std::endl;
            }
            facet.annotation.render(&facet.image, opt);
            facet.annotation.shapes.clear();
            return 0;
        }
    };

    class Round: public Transform {
        int mod;
    public:
        Round (json const &spec) {
        }
        virtual size_t apply_one (Facet *, void const *) const {
            return 0;
        }
    };

    class Clip: public Transform {
    public:
        Clip (json const &spec) {
            CHECK(false);
        }
    };

    class Crop: public Transform {
    public:
        Crop (json const &spec) {
            CHECK(false);
        }
    };

    class ColorSpace: public Transform {
    public:
        ColorSpace (json const &spec) {
            CHECK(false);
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
            if (facet->annotation.shapes.size()) {
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
        std::uniform_real_distribution<float> linear_scale;
        struct PerturbVector {
            float scale;
        };
    public:
        AugScale (json const &spec): linear_scale(spec.value("min", 1.0), spec.value("max", 1.0)) {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->scale = const_cast<AugScale *>(this)->linear_scale(rng);
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, void const *buf) const {
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            if (facet->image.data) {
                cv::Size sz = facet->image.size();
                sz.width = std::round(sz.width * pv->scale);
                sz.height = std::round(sz.height * pv->scale);
                cv::resize(facet->image, facet->image, sz);
            }
            if (facet->annotation.shapes.size()) {
                cv::Size sz = facet->annotation.size;
                sz.width = std::round(sz.width * pv->scale);
                sz.height = std::round(sz.height * pv->scale);
                float fx = 1.0 * sz.width / facet->annotation.size.width;
                float fy = 1.0 * sz.height / facet->annotation.size.height;

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
    public:
        AugAdd (json const &spec) {
            CHECK(false);
        }
    };

    class AugRotate: public Transform {
        std::uniform_real_distribution<float> linear_angle;
        struct PerturbVector {
            float angle;
        };
    public:
        AugRotate (json const &spec): linear_angle(spec.value("min", 0), spec.value("max", 0)) {
        }
        virtual size_t pv_size () const { return sizeof(PerturbVector); }
        virtual size_t pv_sample (random_engine &rng, void *buf) const {
            PerturbVector *pv = reinterpret_cast<PerturbVector *>(buf);
            pv->angle = const_cast<AugRotate *>(this)->linear_angle(rng);
            return sizeof(PerturbVector);
        }
        virtual size_t apply_one (Facet *facet, void const *buf) const {
            PerturbVector const *pv = reinterpret_cast<PerturbVector const *>(buf);
            if (facet->image.data) {
                cv::Size sz = facet->image.size();
                cv::Mat rot = cv::getRotationMatrix2D(cv::Point(sz.width/2, sz.height/2), pv->angle, 1.0);
                cv::Mat tmp;
                cv::warpAffine(facet->image, tmp, rot, sz, facet->type == Facet::LABEL ? CV_INTER_NN: CV_INTER_LINEAR);//, CV_INTER);
                //cv::resize(tmp, tmp, cv::Size(), p.scale, p.scale);
                facet->image = tmp;
            }
            if (facet->annotation.shapes.size()) {
                cv::Size sz = facet->annotation.size;
                cv::Mat rot = cv::getRotationMatrix2D(cv::Point(sz.width/2, sz.height/2), pv->angle, 1.0);
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

    
    std::unique_ptr<Transform> Transform::create (json const &spec) {
        string type = spec.at("class").get<string>();
        if (type == "rasterize") {
            return std::unique_ptr<Transform>(new Rasterize(spec));
        }
        else if (type == "round") {
            return std::unique_ptr<Transform>(new Round(spec));
        }
        else if (type == "clip") {
            return std::unique_ptr<Transform>(new Clip(spec));
        }
        else if (type == "crop") {
            return std::unique_ptr<Transform>(new Crop(spec));
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
        else {
            CHECK(0) << "unknown shape: " << type;
        }
        return nullptr;
    }


}
