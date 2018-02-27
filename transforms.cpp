#include "picpac-image.h"

namespace picpac {

    class Rasterize: public Transform {
    public:
        Rasterize (json const &spec) {
        }
        virtual size_t pv_size () const { return 0; }
        virtual size_t pv_sample (random_engine &, void *) const { return 0; }
        virtual size_t apply (Sample *sample, void *) const {
            return 0;
        }
    };

    class Round: public Transform {
        int mod;
    public:
        Round (json const &spec) {
        }
        virtual size_t pv_size () const { return 0; }
        virtual size_t pv_sample (random_engine &, void *) const { return 0; }
        virtual size_t apply_one (AnnotatedImage *, void *) const {
            return 0;
        }
    };

    class Clip: public Transform {
    public:
        Clip (json const &spec) {
        }
    };

    class Crop: public Transform {
    public:
        Crop (json const &spec) {
        }
    };

    class ColorSpace: public Transform {
    public:
        ColorSpace (json const &spec) {
        }
    };


    class AugFlip: public Transform {
    public:
        AugFlip (json const &spec) {
        }
    };

    class AugScale: public Transform {
    public:
        AugScale (json const &spec) {
        }
    };

    class AugAdd: public Transform {
    public:
        AugAdd (json const &spec) {
        }
    };

    class AugRotate: public Transform {
    public:
        AugRotate (json const &spec) {
        }
    };

    class AugAffine: public Transform {
    public:
        AugAffine (json const &spec) {
        }
    };

    
    std::unique_ptr<Transform> Transform::create (json const &spec) {
        string type = spec.at("type").get<string>();
        if (type == "round") {
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
