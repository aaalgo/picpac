#include <random>
#include <boost/core/noncopyable.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace picpac {

    class PointCloudMapper {
        float scale_u;  // horizontal
        float scale_v;  // vertical
        float min_x, max_x;
        float min_y, max_y;
        float min_z, max_z;
        int U, V;
    public:
        PointCloudMapper ()
            : scale_u(800),
              scale_v(800),
              min_x(0), max_x(80),
              min_y(-50), max_y(50),
              min_z(-100), max_z(100) {
            U = int(std::round(2 * scale_u));
            V = int(std::round(0.5 * scale_v));
        }

        void apply (cv::Mat points,
                    cv::Mat boxes,
                    cv::Mat *map_v,
                    cv::Mat *map_xyz, 
                    cv::Mat *map_mask,
                    cv::Mat *box_params,
                    cv::Mat *box_mask      // 0 / 1
                ) const {
            *map_v = cv::Mat(V, U, CV_32FC1, cv::Scalar(0));
            *map_xyz = cv::Mat(V, U, CV_32FC3, cv::Scalar(0,0,0));
            *map_mask = cv::Mat(V, U, CV_32FC1, cv::Scalar(0));
            *box_params = cv::Mat(V, U, CV_32FC(6), cv::Scalar(0));
            *box_mask = cv::Mat(V, U, CV_32FC1, cv::Scalar(0));

            CHECK(points.cols == 4);
            CHECK(boxes.cols == 6);
            for (int i = 0; i < points.rows; ++i) {
                float const *p = points.ptr<float const>(i);
                if (!((p[0] >= min_x) && (p[0] <= max_x)
                    && (p[1] >= min_y) && (p[1] <= max_y)
                    && (p[2] >= min_z) && (p[2] <= max_z))) continue;
                float r = sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
                float sin_u = p[1] / r; // left-right
                float sin_v = p[2] / r; // bottom-top

                int u = int(std::round((asin(sin_u) + 1) * scale_u));
                int v = int(std::round(V - (asin(sin_v) + 0.375) * scale_v));

                if (u < 0 || u >= U) continue;
                if (v < 0 || v >= V) continue;

                {
                    float *t = map_v->ptr<float>(v) + u;
                    *t = p[3];
                }
                {
                    float *t = map_xyz->ptr<float>(v) + 3 * u;
                    t[0] = p[0];
                    t[1] = p[1];
                    t[2] = p[2];
                }
                {
                    float *t = map_mask->ptr<float>(v) + u;
                    *t = 1;
                }
                for (int j = 0;  j < boxes.rows; ++j) {
                    // set to boxes
                    float const *b = boxes.ptr<float const>(j);
                    if   ((p[0] >= b[0]) && (p[0] <= b[1])  // inside box
                       && (p[1] >= b[2]) && (p[1] <= b[3])
                       && (p[2] >= b[4]) && (p[2] <= b[5])) {
                        // good
                        *(box_mask->ptr<float>(v) + u) = 1;
                        float *t = box_params->ptr<float>(v) + u * 6;

                        t[0] = (b[0] + b[1]) / 2 - p[0];
                        t[1] = (b[2] + b[3]) / 2 - p[1];
                        t[2] = (b[4] + b[5]) / 2 - p[2];
                        t[3] = (b[1] - b[0]);
                        t[4] = (b[3] - b[2]);
                        t[5] = (b[5] - b[4]);
                        break;
                    }
                }
            }
        }
    };
}
