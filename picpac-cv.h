#pragma once
#include <random>
#include <opencv2/opencv.hpp>
#include "picpac.h"

namespace picpac {

    class ImageLoader {
    public:
        enum {
            ANNOTATE_NONE = 0,
            ANNOTATE_LOAD = 1,
            ANNOTATE_JSON = 2
        };
        struct Config {
            cv::Size resize;
            int mode;
            int annotate;
            int anno_type;
            cv::Scalar anno_color;
            int anno_thickness;
            bool perturb;
            cv::Scalar pert_color;
            float pert_angle;
            float pert_scale;
            bool pert_hflip, pert_vflip;
            Config ()
                : resize(0, 0), // do not resize by default
                mode(cv::IMREAD_UNCHANGED),
                annotate(ANNOTATE_NONE),
                anno_type(CV_8UC1),
                anno_color(1),
                anno_thickness(CV_FILLED),
                perturb(false),
                pert_color(0,0,0),
                pert_angle(0),
                pert_scale(0),
                pert_hflip(false),
                pert_vflip(false) {
            }
        };

        struct Value {
            float label;
            cv::Mat image;
            cv::Mat annotation;
        };

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
            linear_scale(-c.pert_scale, c.pert_scale)
        {
        }

        template <typename RNG>
        void sample (RNG &e, PerturbVector *p) {
            if (config.perturb) {
                p->hflip = (e() % 2) & config.pert_hflip;
                p->vflip = (e() % 2) & config.pert_vflip;
                p->color[0] = delta_color1(e);
                p->color[1] = delta_color2(e);
                p->color[2] = delta_color3(e);
                p->angle = linear_angle(e);
                p->scale = std::exp(linear_scale(e));
            }
        }

        void load (Record &&in, PerturbVector const &p, Value *out) const; 

    private:
        Config config;
        std::uniform_int_distribution<int> delta_color1; //(min_R, max_R);
        std::uniform_int_distribution<int> delta_color2; //(min_R, max_R);
        std::uniform_int_distribution<int> delta_color3; //(min_R, max_R);
        std::uniform_real_distribution<float> linear_angle;
        std::uniform_real_distribution<float> linear_scale;
    };

    class ImageReader {
        int max;
        int mode;
        string code;
    public:
        ImageReader (int max_ = 800, int mode_ = cv::IMREAD_UNCHANGED, string code_ = string())
            : max(max_), mode(mode_), code(code_) {
        }

        void read (fs::path const &path, string *data);
    };
}

