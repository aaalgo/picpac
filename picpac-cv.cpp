#include <json11.hpp>
#include <boost/filesystem/fstream.hpp>
#include "picpac-cv.h"

namespace picpac {

    using namespace json11;

    static float LimitSize (cv::Mat input, int max_size, cv::Mat *output) {
        if (input.rows == 0) {
            *output = cv::Mat();
            return 0;
        }
        float scale = 1.0;
        int maxs = std::max(input.cols, input.rows);
        // large side > max
        if ((max_size > 0) && (maxs > max_size)) {
            cv::Mat tmp;
            scale = 1.0 * maxs / max_size;
            cv::resize(input, tmp, cv::Size(input.cols * max_size / maxs, input.rows * max_size / maxs));
            input = tmp;
        }
        *output = input;
        return scale;
    }


    class Shape {
    public:
        virtual void draw (cv::Mat *, cv::Scalar v, int thickness = CV_FILLED) const = 0;
        static std::shared_ptr<Shape> create (Json const &geo);
    };

    class Box: public Shape {
        cv::Rect_<float> rect;
    public:
        Box (Json const &geo) {
            rect.x = geo["x"].number_value();
            rect.y = geo["y"].number_value();
            rect.width = geo["width"].number_value();
            rect.height = geo["height"].number_value();
        }
        virtual void draw (cv::Mat *m, cv::Scalar v, int thickness) const {
            cv::Rect box;
            box.x = std::round(m->cols * rect.x);
            box.y = std::round(m->rows * rect.y);
            box.width = std::round(m->cols * rect.width);
            box.height = std::round(m->rows * rect.height);
            cv::rectangle(*m, box, v, thickness);
        }
    };

    class Ellipse: public Shape {
        cv::Rect_<float> rect;
    public:
        Ellipse (Json const &geo) {
            rect.x = geo["x"].number_value();
            rect.y = geo["y"].number_value();
            rect.width = geo["width"].number_value();
            rect.height = geo["height"].number_value();
        }
        virtual void draw (cv::Mat *m, cv::Scalar v, int thickness) const {
            cv::Point2f center(m->cols * (rect.x + rect.width/2),
                               m->rows * (rect.y + rect.height/2));
            cv::Size2f size(m->cols * rect.width, m->rows * rect.height);
            cv::ellipse(*m, cv::RotatedRect(center, size, 0), v, thickness);
        }
    };

    class Poly: public Shape {
        vector<cv::Point2f> points;
    public:
        Poly (Json const &geo) {
            for (auto const &p: geo["points"].array_items()) {
                points.emplace_back(p["x"].number_value(), p["y"].number_value());
            }
        }
        virtual void draw (cv::Mat *m, cv::Scalar v, int thickness) const {
            vector<cv::Point> ps(points.size());
            for (unsigned i = 0; i < ps.size(); ++i) {
                auto const &from = points[i];
                auto &to = ps[i];
                to.x = std::round(from.x * m->cols);
                to.y = std::round(from.y * m->rows);
            }
            cv::Point const *pps = &ps[0];
            int const nps = ps.size();
            if (thickness == CV_FILLED) {
                cv::fillPoly(*m, &pps, &nps, 1, v);
            }
            else {
                cv::polylines(*m, &pps, &nps, 1, true, v, thickness);
            }
        }
    };

    std::shared_ptr<Shape> Shape::create (Json const &geo) {
        string type = geo["type"].string_value();
        if (type == "rect") {
            return std::shared_ptr<Shape>(new Box(geo["geometry"]));
        }
        if (type == "ellipse") {
            return std::shared_ptr<Shape>(new Ellipse(geo["geometry"]));
        }
        else if (type == "polygon") {
            return std::shared_ptr<Shape>(new Poly(geo["geometry"]));
        }
        CHECK(0) << "unknown shape: " << type;
        return 0;
    }

    class Annotation {
        vector<std::shared_ptr<Shape>> shapes;
    public:
        Annotation () {}
        Annotation (string const &txt) {
            string err;
            Json json = Json::parse(txt, err);
            if (err.size()) {
                LOG(ERROR) << "Bad json: " << err << " (" << txt << ")";
                return;
            }
            for (auto const &x: json["shapes"].array_items()) {
                shapes.emplace_back(Shape::create(x));
            }
        }

        void draw (cv::Mat *m, cv::Scalar v, int thickness = -1) const {
            for (auto const &p: shapes) {
                p->draw(m, v, thickness);
            }
        }
    };

    void ImageLoader::load (RecordReader rr, PerturbVector const &p, Value *out,
           CacheValue *cache, std::mutex *mutex) const {
        Value cached;
        do {
            if (cache) { // access cache
                lock_guard lock(*mutex);
                cached = *cache;
            }
            if (cached.image.data) break;
            // cache miss, load the data
            Record r;
            rr(&r); // disk access
            cached.label = r.meta().label;
            CHECK(r.size() >= (annotate ? 2 : 1));
            auto imbuf = r.field(0);
            cached.image = cv::imdecode(cv::Mat(1, boost::asio::buffer_size(imbuf), CV_8U,
                                const_cast<void *>(boost::asio::buffer_cast<void const *>(imbuf))),
                                config.decode_mode);
            if ((config.channels > 0) && config.channels != cached.image.channels()) {
                cv::Mat tmp;
                if (cached.image.channels() == 3 && config.channels == 1) {
                    cv::cvtColor(cached.image, tmp, CV_BGR2GRAY);
                }
                else if (cached.image.channels() == 4 && config.channels == 1) {
                    cv::cvtColor(cached.image, tmp, CV_BGRA2GRAY);
                }
                else if (cached.image.channels() == 4 && config.channels == 3) {
                    cv::cvtColor(cached.image, tmp, CV_BGRA2BGR);
                }
                else if (cached.image.channels() == 1 && config.channels == 3) {
                    cv::cvtColor(cached.image, tmp, CV_GRAY2BGR);
                }
                else CHECK(0) << "channel format not supported: from "
                              << cached.image.channels()
                              << " to " << config.channels;
                cached.image = tmp;
            }
            if (config.resize_width > 0 && config.resize_height > 0) {
                cv::resize(cached.image, cached.image, cv::Size(config.resize_width, config.resize_height), 0, 0);
            }
            else if (config.max_size > 0) {
                cv::Mat tmp;
                LimitSize(cached.image, config.max_size, &tmp);
                cached.image = tmp;
            }
            if (annotate == ANNOTATE_IMAGE) {
                auto anbuf = r.field(1);
                cached.annotation = cv::imdecode(cv::Mat(1, boost::asio::buffer_size(anbuf), CV_8U,
                                const_cast<void *>(boost::asio::buffer_cast<void const *>(anbuf))),
                                cv::IMREAD_UNCHANGED);
                if (cached.annotation.size() != cached.image.size()) {
                    cv::resize(cached.annotation, cached.annotation, cached.image.size(), 0, 0, cv::INTER_NEAREST);
                }
            }
            else if (annotate == ANNOTATE_JSON) {
                Annotation a(r.field_string(1));
                cv::Mat anno;
                if (config.anno_copy) {
                    anno = cached.image.clone();
                }
                else {
                    anno = cv::Mat(cached.image.size(), config.anno_type, cv::Scalar(0));
                }
                cv::Scalar color(config.anno_color1,
                             config.anno_color2,
                             config.anno_color3);
                a.draw(&anno, color, config.anno_thickness);
                cached.annotation = anno;
            }
            if (cache) {
                // write to cache
                lock_guard lock(*mutex);
                *cache = cached;
            }
        } while (false);

        if (!config.perturb) {
            *out = cached;
            return;
        }

        //float color, angle, scale, flip = false;
        //cv::Size sz = cached.image.size();
        cv::Mat image = cached.image, anno = cached.annotation;
        
        cv::Mat rot = cv::getRotationMatrix2D(cv::Point(image.cols/2, image.rows/2), p.angle, p.scale);
        {
            cv::Mat tmp;
            cv::warpAffine(image, tmp, rot, image.size(), cv::INTER_LINEAR, config.pert_border);
            //cv::resize(tmp, tmp, cv::Size(), p.scale, p.scale);
            image = tmp;
        }

        if (cached.annotation.data) {
            //cv::resize(anno, anno, cv::Size(), p.scale, p.scale, cv::INTER_NEAREST);
            cv::Mat tmp;
            cv::warpAffine(anno, tmp, rot, anno.size(), cv::INTER_NEAREST, config.pert_border); // cannot interpolate labels
            anno = tmp;
        }

        image += p.color;

        if (p.hflip && p.vflip) {
            cv::flip(image, out->image, -1);
            cv::flip(anno, out->annotation, -1);
        }
        else if (p.hflip && !p.vflip) {
            cv::flip(image, out->image, 1);
            cv::flip(anno, out->annotation, 1);
        }
        else if (!p.hflip && p.vflip) {
            cv::flip(image, out->image, 0);
            cv::flip(anno, out->annotation, 0);
        }
        else {
            out->image = image;
            out->annotation = anno;
        }
        out->label = cached.label;
    }

    /*
    int prod (vector<int> const &v) {
        int r = 1;
        for (auto const &x: v) {
            r *= x;
        }
        return r;
    }
    */

    void ImageEncoder::encode (cv::Mat const &image, string *data) {
        std::vector<uint8_t> buffer;
        cv::imencode(code.empty() ? ".jpg": code, image, buffer);
        char const *from = reinterpret_cast<char const *>(&buffer[0]);
        *data = string(from, from + buffer.size());
    }

    void ImageReader::read (fs::path const &path, string *data) {
        bool do_code = code.size() || (mode != cv::IMREAD_UNCHANGED);
        cv::Mat image = cv::imread(path.native(), mode);
        if (!image.data) throw BadFile(path);
        if (resize > 0) {
            cv::resize(image, image, cv::Size(resize, resize));
            do_code = true;
        }
        else if (max > 0) {
            cv::Mat rs;
            LimitSize(image, max, &rs);
            if (rs.total() != image.total()) {
                image = rs;
                do_code = true;
            }
        }
        if (do_code) {
            encode(image, data);
        }
        else {
            // read original file
            uintmax_t sz = fs::file_size(path);
            data->resize(sz);
            fs::ifstream is(path, std::ios::binary);
            is.read(&data->at(0), data->size());
            if (!is) throw BadFile(path);
        }
    }
}

