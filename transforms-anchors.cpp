#include "picpac-image.h"

namespace picpac {

    class Anchors: public Transform {
        int index;          // 
        int downsize;
        float upper_th;
        float lower_th;
        vector<float> ratios;
        vector<float> sizes;

        struct Box {
            float x1, y1, x2, y2;
            float score;
            uint8_t *label;
            float *label_mask;
            float *params;
            cv::Point2f shift;
        };

    public:
        Anchors (json const &spec) {
            index = spec.value<int>("index", 1);
            downsize = spec.value<int>("downsize", 1);
            upper_th = spec.value<float>("upper_th", 0.8);
            lower_th = spec.value<float>("lower_th", 0.4);
            float size = spec.value<int>("size", 30);
            
            if (spec.find("ratios") == spec.end()) {
                ratios.push_back(1.0);
            }
            else 
            for (auto const &r: spec.at("ratios")) {
                ratios.push_back(r.get<float>());
            }
            if (spec.find("sizes") == spec.end()) {
            }
            else 
            for (auto const &r: spec.at("sizes")) {
                ratios.push_back(r.get<float>());
            }
        }

        virtual size_t apply (Sample *sample, void const *) const {
#if 0
            auto const &facet = sample->facets[index];
            auto const &anno = facet.annotation;

            CHECK(!anno.empty());

            vector<Box> truths(anno.shapes.size());  // ground-truth circles
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
                c.x1 = minx;
                c.x2 = maxx;
                c.y1 = miny;
                c.y2 = maxy;

                c.score = numeric_limits<float>::max();
                c.label = nullptr;
                c.label_mask = nullptr;
                c.params = nullptr;
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
            /*
            sample->facets.emplace_back(params);
            sample->facets.emplace_back(params_mask);
            */
#endif
            return 0;
        }
    };

    Transform *create_anchors (json const &spec) {
        return new Anchors(spec);
    }

}
