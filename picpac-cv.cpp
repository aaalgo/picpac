#include <json11.hpp>
#include <boost/filesystem/fstream.hpp>
#include "picpac-cv.h"

namespace picpac {

    using namespace json11;

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
        Annotation (const_buffer const &buf) {
            char const *begin = boost::asio::buffer_cast<char const *>(buf);
            auto sz = boost::asio::buffer_size(buf);
            string txt(begin, begin + sz);
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


    void ImageLoader::load (Record &&in, PerturbVector const &p, Value *out) const {
        out->label = in.meta().label;
        CHECK(in.size() >= (config.annotate ? 2 : 1));
        auto imbuf = in.field(0);
        cv::Mat image = cv::imdecode(cv::Mat(1, boost::asio::buffer_size(imbuf), CV_8U,
                            const_cast<void *>(boost::asio::buffer_cast<void const *>(imbuf))),
                            config.mode);
        cv::Mat anno;
        if (config.annotate == ANNOTATE_LOAD) {
            auto anbuf = in.field(1);
            anno = cv::imdecode(cv::Mat(1, boost::asio::buffer_size(anbuf), CV_8U,
                            const_cast<void *>(boost::asio::buffer_cast<void const *>(anbuf))),
                            cv::IMREAD_UNCHANGED);
        }
        else if (config.annotate == ANNOTATE_JSON) {
            auto anbuf = in.field(1);
            Annotation a(anbuf);
            anno = cv::Mat(image.size(), config.anno_type, cv::Scalar(0));
            a.draw(&anno, config.anno_color, config.anno_thickness);
        }
        if (config.resize.width) {
            cv::resize(image, image, config.resize, 0, 0);
            if (anno.data) {
                cv::resize(anno, anno, config.resize, 0, 0, cv::INTER_NEAREST);
            }
        }
        out->image = image;
        out->annotation = anno;
    }

    static float LimitSize (cv::Mat input, int max_size, cv::Mat *output) {
        if (input.rows == 0) {
            *output = cv::Mat();
            return 0;
        }
        float scale = 1.0;
        int maxs = std::max(input.cols, input.rows);
        // large side > max
        if (maxs > max_size) {
            cv::Mat tmp;
            scale = 1.0 * maxs / max_size;
            cv::resize(input, tmp, cv::Size(input.cols * max_size / maxs, input.rows * max_size / maxs));
            input = tmp;
        }
        *output = input;
        return scale;
    }

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
        if (max > 0) {
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

