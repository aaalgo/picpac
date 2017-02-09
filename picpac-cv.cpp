#include <iostream>
#include <sstream>
#include <json11.hpp>
#include <boost/filesystem/fstream.hpp>
#include "picpac-cv.h"

namespace picpac {

    using namespace json11;

    float LimitSize (cv::Mat input, int min_size, int max_size, cv::Mat *output) {
        if (input.rows == 0) {
            *output = cv::Mat();
            return 0;
        }
        float scale = 1.0;
        int maxs = std::max(input.cols, input.rows);
        int mins = std::min(input.cols, input.rows);

        if ((max_size > 0) && (maxs > max_size)) {
            cv::Mat tmp;
            scale = 1.0 * maxs / max_size;
            cv::resize(input, tmp, cv::Size(input.cols * max_size / maxs, input.rows * max_size / maxs));
            input = tmp;
        }
        // large side > max
        if ((min_size > 0) && (mins < min_size)) {
            cv::Mat tmp;
            scale = 1.0 * min_size / mins;
            cv::resize(input, tmp, cv::Size(input.cols * min_size / mins, input.rows * min_size / mins));
            input = tmp;
        }
        *output = input;
        return scale;
    }

    class Box: public Shape {
    protected:
        cv::Rect_<float> rect;
    public:
        Box (Json const &geo, char const *t = "rect"): Shape(t) {
            rect.x = geo["x"].number_value();
            rect.y = geo["y"].number_value();
            rect.width = geo["width"].number_value();
            rect.height = geo["height"].number_value();
        }
        virtual void dump (Json *json) const {
            *json = Json::object {
                {"type", type()},
                {"geometry", Json::object{
                                      {"x", rect.x},
                                      {"y", rect.y},
                                      {"width", rect.width},
                                      {"height", rect.height}
                                   }}
            };
        }
        virtual std::shared_ptr<Shape> clone () const {
            return std::shared_ptr<Shape>(new Box(*this));
        }
        virtual void draw (cv::Mat *m, cv::Scalar v, int thickness) const {
            cv::Rect box;
            box.x = std::round(m->cols * rect.x);
            box.y = std::round(m->rows * rect.y);
            box.width = std::round(m->cols * rect.width);
            box.height = std::round(m->rows * rect.height);
            cv::rectangle(*m, box, v, thickness);
        }
        virtual void bbox (cv::Rect_<float> *bb) const {
            *bb = rect;
        }
        virtual void zoom (cv::Rect_<float> const &bb) {
            rect.x -= bb.x;
            rect.y -= bb.y;
            rect.x /= bb.width;
            rect.width /= bb.width;
            rect.y /= bb.height;
            rect.height /= bb.height;
        }
    };

    class Ellipse: public Box {
    public:
        Ellipse (Json const &geo): Box(geo, "ellipse") {
        }
        virtual std::shared_ptr<Shape> clone () const {
            return std::shared_ptr<Shape>(new Ellipse(*this));
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
        Poly (Json const &geo): Shape("polygon") {
            for (auto const &p: geo["points"].array_items()) {
                points.emplace_back(p["x"].number_value(), p["y"].number_value());
            }
        }
        virtual void dump (Json *json) const {
            vector<Json> pts;
            for (auto const &p: points) {
                pts.emplace_back(Json::object{{"x", p.x}, {"y", p.y}});
            }
            *json = Json::object {
                {"type", type()},
                {"geometry", Json::object{{"points", std::move(pts)}}}
            };
        }
        virtual std::shared_ptr<Shape> clone () const {
            return std::shared_ptr<Poly>(new Poly(*this));
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
        virtual void bbox (cv::Rect_<float> *bb) const {
            float min_x = 1, min_y = 1;
            float max_x = 0, max_y = 0;
            for (auto const &p: points) {
                if (p.x < min_x) min_x = p.x;
                if (p.y < min_y) min_y = p.y;
                if (p.x > max_x) max_x = p.x;
                if (p.y > max_y) max_y = p.y;
            }
            float area = (max_x - min_x) * (max_y - min_y);
            if (area <= 0) {
                *bb = cv::Rect_<float>();
            }
            else {
                bb->x = min_x;
                bb->y = min_y;
                bb->width = max_x - min_x;
                bb->height = max_y - min_y;
            }
        }
        virtual void zoom (cv::Rect_<float> const &bb) {
            for (auto &p: points) {
                p.x -= bb.x;
                p.y -= bb.y;
                p.x /= bb.width;
                p.y /= bb.height;
            }
        }
    };

    std::shared_ptr<Shape> Shape::create (Json const &geo) {
        string type = geo["type"].string_value();
        std::shared_ptr<Shape> shape;
        if (type == "rect") {
            shape = std::shared_ptr<Shape>(new Box(geo["geometry"]));
        }
        if (type == "ellipse") {
            shape = std::shared_ptr<Shape>(new Ellipse(geo["geometry"]));
        }
        else if (type == "polygon") {
            shape = std::shared_ptr<Shape>(new Poly(geo["geometry"]));
        }
        else {
            CHECK(0) << "unknown shape: " << type;
        }
        Json const &label = geo["label"];
        if (label.is_array()) {
            cv::Scalar ll(0,0,0);
            auto const &array = label.array_items();
            CHECK(array.size() <= 3) << "can only support label of <= 3 channels";
            for (unsigned i = 0; i < array.size(); ++i) {
                ll[i] = array[i].number_value();
            }
            shape->setLabel(ll);
        }
        else {
            shape->setLabel(cv::Scalar(label.number_value(), 0,0));
        }
        return 0;
    }

    Annotation::Annotation (string const &txt) {
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

    void Annotation::dump (string *str) const {
        vector<Json> ss;
        for (auto const &p: shapes) {
            ss.emplace_back();
            p->dump(&ss.back());
        }
        Json json = Json::object {{"shapes", std::move(ss)}};
        *str = json.dump();
    }

    void Annotation::draw (cv::Mat *m, cv::Scalar v, int thickness) const {
        for (auto const &p: shapes) {
            if (p->haveLabel()) {
                v = p->label();
            }
            p->draw(m, v, thickness);
        }
    }

    void  Annotation::bbox (cv::Rect_<float> *bb) const { float min_x = 1, min_y = 1;
        float max_x = 0, max_y = 0;
        for (auto const &p: shapes) {
            cv::Rect_<float> sb;
            p->bbox(&sb);
            if (sb.area() <= 0) continue;

            if (sb.x < min_x) min_x = sb.x;
            if (sb.y < min_y) min_y = sb.y;

            sb.x += sb.width;
            sb.y += sb.height;

            if (sb.x > max_x) max_x = sb.x;
            if (sb.y > max_y) max_y = sb.y;
        }
        float area = (max_x - min_x) * (max_y - min_y);
        if (area <= 0) {
            *bb = cv::Rect_<float>();
        }
        else {
            bb->x = min_x;
            bb->y = min_y;
            bb->width = max_x - min_x;
            bb->height = max_y - min_y;
        }
    }

    void Annotation::zoom (cv::Rect_<float> const &bb) {
        for (auto &p: shapes) {
            p->zoom(bb);
        }
    }

    // for visualization only!!!
    void spectrogram_to_gray (cv::Mat m, cv::Mat *gray) {
        cv::Mat o(m.rows, m.cols, CV_32F);
        for (int i = 0; i < o.rows; ++i) {
            float const *in = m.ptr<float const>(i);
            float *out = o.ptr<float>(i);
            for (int j = 0; j < o.cols; ++j) {
                out[0] = std::sqrt(in[0] * in[0]
                                 + in[1] * in[1]);
                in += 2;
                out += 1;
            }
        }
        cv::normalize(o, o, 0, 255, cv::NORM_MINMAX, CV_8U);
        *gray = o;
    }

    // for visualization only!!!
    void spectrogram_to_bgr (cv::Mat m, cv::Mat *bgr) {
        float mm = 0;
        for (int i = 0; i < m.rows; ++i) {
            float const *in = m.ptr<float const>(i);
            for (int j = 0; j < m.cols; ++j) {
                float a = in[0];
                float b = in[1];
                float v = std::sqrt(a*a + b*b);
                if (v > mm) mm = v;
                in += 2;
            }
        }
        cv::Mat o(m.rows, m.cols, CV_8UC3);
        static const float BASE = 0.005;
        for (int i = 0; i < o.rows; ++i) {
            float const *in = m.ptr<float const>(i);
            uint8_t *out = o.ptr<uint8_t>(i);
            for (int j = 0; j < o.cols; ++j) {
                float a = in[0];
                float b = in[1];
                float v = std::sqrt(a*a + b*b)/mm;
                v = (log(std::max(BASE, v)) - log(BASE))/(-log(BASE)) ;
                out[0] = (std::atan2(b, a) + M_PI)/M_PI * 90;
                out[1] = 255;
                out[2] = BASE + (255-BASE) * v;
                in += 2;
                out += 3;
            }
        }
        cv::cvtColor(o, *bgr, CV_HSV2BGR);
    }

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
            //CHECK(r.size() >= (annotate ? 2 : 1));
            cached.image = decode_buffer(r.field(0), config.decode_mode);
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
#ifdef SUPPORT_AUDIO_SPECTROGRAM
                else if (cached.image.type() == CV_32FC2 && config.channels == 1) {
                    spectrogram_to_gray(cached.image, &tmp);
                }
                else if (cached.image.type() == CV_32FC2 && config.channels == 3) {
                    spectrogram_to_bgr(cached.image, &tmp);
                }
#endif
                else CHECK(0) << "channel format not supported: from "
                              << cached.image.channels()
                              << " to " << config.channels;
                cached.image = tmp;
            }
            if (config.resize_width > 0 && config.resize_height > 0) {
                cv::resize(cached.image, cached.image, cv::Size(config.resize_width, config.resize_height), 0, 0);
            }
            else if (config.max_size > 0 || config.min_size > 0) {
                cv::Mat tmp;
                LimitSize(cached.image, config.min_size, config.max_size, &tmp);
                cached.image = tmp;
            }
            if (annotate == ANNOTATE_IMAGE) {
                if (r.size() < 2) {
                    cached.annotation = cv::Mat(cached.image.size(), CV_8U, cv::Scalar(0));
                }
                else {
                    auto anbuf = r.field(1);
                    cached.annotation = cv::imdecode(cv::Mat(1, boost::asio::buffer_size(anbuf), CV_8U,
                                const_cast<void *>(boost::asio::buffer_cast<void const *>(anbuf))),
                                cv::IMREAD_UNCHANGED);
                    if (!cached.annotation.data) {
                        cached.annotation = decode_raw(boost::asio::buffer_cast<char const *>(anbuf),
                                            boost::asio::buffer_size(anbuf));
                    }
                    if (cached.annotation.size() != cached.image.size()) {
                        cv::resize(cached.annotation, cached.annotation, cached.image.size(), 0, 0, cv::INTER_NEAREST);
                    }
                }
                CHECK(config.anno_min_ratio == 0) << "Not supported";
            }
            else if (annotate == ANNOTATE_JSON) {
                cv::Mat anno;
                if (config.anno_copy) {
                    anno = cached.image.clone();
                }
                else {
                    anno = cv::Mat(cached.image.size(), config.anno_type, cv::Scalar(0));
                }
                if (r.size() > 1) {
                    Annotation a(r.field_string(1));
                    cv::Scalar color(config.anno_color1,
                                 config.anno_color2,
                                 config.anno_color3);
                    a.draw(&anno, color, config.anno_thickness);

                    // we might want to perturb the cropping
                    // this might not be the perfect location either
                    do {
                        if (!(config.anno_min_ratio > 0)) break;
                        if (!(anno.total() > 0)) break;
                        cv::Rect_<float> fbb;
                        a.bbox(&fbb);
                        if (!(fbb.area() > 0)) break;
                        fbb.x *= anno.cols;
                        fbb.width *= anno.cols;
                        fbb.y *= anno.rows;
                        fbb.height *= anno.rows;

                        float min_roi_size = anno.total() * config.anno_min_ratio;
                        float roi_size = fbb.area();
                        if (roi_size >= min_roi_size) break;
                        float rate = std::sqrt(roi_size / min_roi_size);
                        int width = std::round(anno.cols * rate);
                        int height = std::round(anno.rows * rate);
                        cv::Rect bbox;
                        bbox.x = std::round(fbb.x);
                        bbox.y = std::round(fbb.y);
                        bbox.width = std::round(fbb.width);
                        bbox.height = std::round(fbb.height);
                        int dx = (width - bbox.width)/2;
                        int dy = (height - bbox.height)/2;
                        bbox.x -= dx;
                        bbox.y -= dy;
                        bbox.width = width;
                        bbox.height = height;
                        if (bbox.x < 0) {
                            bbox.x = 0;
                        }
                        if (bbox.y < 0) {
                            bbox.y = 0;
                        }
                        if (bbox.x + bbox.width > anno.cols) {
                            bbox.x = anno.cols - bbox.width;
                        }
                        if (bbox.y + bbox.height > anno.rows) {
                            bbox.y = anno.rows - bbox.height;
                        }
                        cv::Mat im = cached.image(bbox);
                        cv::Mat an = anno(bbox);
                        cv::Mat rim, ran;
                        cv::resize(im, rim, cached.image.size());
                        cv::resize(an, ran, anno.size());
                        cached.image = rim;
                        anno = ran;
                    } while (false);

                }
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
            if ((config.crop_width > 0) && (config.crop_height > 0)) {
                CHECK(out->image.cols >= config.crop_width);
                CHECK(out->image.rows >= config.crop_height);
                // cropping
                int marginx = out->image.cols - config.crop_width;
                int marginy = out->image.rows - config.crop_height;
                cv::Rect roi(marginx / 2,
                             marginy / 2,
                             config.crop_width,
                             config.crop_height);
                out->image = out->image(roi);
                if (out->annotation.data) {
                    out->annotation = out->annotation(roi);
                }
            }
            if (config.round_div > 0) {
                int width = out->image.cols;
                int height = out->image.rows;
                width = width / config.round_div * config.round_div + config.round_mod;
                height = height / config.round_div * config.round_div + config.round_mod;
                if (width > out->image.cols) {
                    width -= config.round_div;
                }
                if (height > out->image.rows) {
                    height -= config.round_div;
                }
                CHECK((width > 0) && (height > 0));
                int marginx = out->image.cols - width;
                int marginy = out->image.rows - height;
                if (!config.perturb) {
                    marginx /= 2;
                    marginy /= 2;
                }
                else {
                    marginx = p.shiftx % (marginx+1);
                    marginy = p.shifty % (marginy+1);
                }
                cv::Rect roi(marginx, marginy, 
                             width, height); 
                out->image = out->image(roi);
                if (out->annotation.data) {
                    out->annotation = out->annotation(roi);
                }
            }
            if (annotate == ANNOTATE_AUTO) {
                out->annotation = out->image;
            }
            return;
        }

        //float color, angle, scale, flip = false;
        //cv::Size sz = cached.image.size();
        cv::Mat image = cached.image, anno = cached.annotation;
        
        //TODO: scale might break min and/or max restriction
        if (p.angle != 0) {
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
        }
        else if (p.scale != 1) {
            {
                cv::Mat tmp;
                cv::resize(image, tmp, cv::Size(), p.scale, p.scale);
                image = tmp;
            }
            if (cached.annotation.data) {
                cv::Mat tmp;
                cv::resize(anno, tmp, cv::Size(), p.scale, p.scale, cv::INTER_NEAREST);
                anno = tmp;
            }
        }

        /*
        std::cout << p.color[0] << " " << p.color[1] << " " << p.color[2] << std::endl;
        std::cout << colorspace << std::endl;
        */
        if (image.channels() == 3) {
            if (colorspace == COLOR_Lab) {
                cv::cvtColor(image, image, CV_BGR2Lab);
            }
            else if (colorspace  == COLOR_HSV) {
                cv::cvtColor(image, image, CV_BGR2HSV);
            }
        }
        image += p.color;
        if (image.channels() == 3) {
            if (colorspace == COLOR_Lab) {
                cv::cvtColor(image, image, CV_Lab2BGR);
            }
            else if (colorspace  == COLOR_HSV) {
                cv::cvtColor(image, image, CV_HSV2BGR);
            }
        }

        if (p.hflip && p.vflip) {
            cv::flip(image, out->image, -1);
            if (anno.data) {
                cv::flip(anno, out->annotation, -1);
            }
        }
        else if (p.hflip && !p.vflip) {
            cv::flip(image, out->image, 1);
            if (anno.data) {
                cv::flip(anno, out->annotation, 1);
            }
        }
        else if (!p.hflip && p.vflip) {
            cv::flip(image, out->image, 0);
            if (anno.data) {
                cv::flip(anno, out->annotation, 0);
            }
        }
        else {
            out->image = image;
            if (anno.data) {
                out->annotation = anno;
            }
        }
        if ((config.crop_width > 0) && (config.crop_height > 0)
            && ((out->image.cols > config.crop_width)
            || (out->image.rows > config.crop_height))) {
            // cropping
            int marginx = out->image.cols - config.crop_width;
            int marginy = out->image.rows - config.crop_height;
            cv::Rect roi(p.shiftx % marginx,
                         p.shifty % marginy,
                         config.crop_width,
                         config.crop_height);
            out->image = out->image(roi);
            if (out->annotation.data) {
                out->annotation = out->annotation(roi);
            }
        }
        if (config.round_div > 0) {
            int width = out->image.cols;
            int height = out->image.rows;
            width = width / config.round_div * config.round_div + config.round_mod;
            height = height / config.round_div * config.round_div + config.round_mod;
            if (width > out->image.cols) {
                width -= config.round_div;
            }
            if (height > out->image.rows) {
                height -= config.round_div;
            }
            CHECK((width > 0) && (height > 0));
            int marginx = out->image.cols - width;
            int marginy = out->image.rows - height;
            if (!config.perturb) {
                marginx /= 2;
                marginy /= 2;
            }
            else {
                marginx = p.shiftx % (marginx+1);
                marginy = p.shifty % (marginy+1);
            }
            cv::Rect roi(marginx, marginy, 
                         width, height); 
            out->image = out->image(roi);
            if (out->annotation.data) {
                out->annotation = out->annotation(roi);
            }
        }
#if 0
        if (out->annotation.data) {
            // replicate positive region
            CHECK(out->annotation.type() == CV_8UC1);
            cv::Rect roi;
            cv::Rect target = roi;
            bound(out->annotation, &roi, 0.98);
            for (unsigned i = 0; i < 10; ++i) {
                for (;;) {
                    target.x = rand() % (out->image.cols - roi.width);
                    target.y = rand() % (out->image.rows - roi.height);
                    if ((target & roi).area() == 0) break;
                }
                out->image(target) = out->image(roi);
                out->annotation(target) = out->annotation(roi);
            }

        }
#endif
        if (annotate == ANNOTATE_AUTO) {
            out->annotation = out->image;
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
        if (code == "raw") {
            encode_raw(image, data);
            return;
        }
        std::vector<uint8_t> buffer;
        cv::imencode(code.empty() ? ".jpg": code, image, buffer);
        char const *from = reinterpret_cast<char const *>(&buffer[0]);
        *data = string(from, from + buffer.size());
    }

    void ImageReader::read (fs::path const &path, string *data) {
        bool do_code = code.size() || (mode != cv::IMREAD_UNCHANGED);
        cv::Mat image = cv::imread(path.native(), mode);
        if (!image.data) { // try raw
            string buf;
            fs::ifstream is(path, std::ios::binary);
            if (!is) throw BadFile(path);
            is.seekg(0, std::ios::end);
            if (!is) throw BadFile(path);
            buf.resize(is.tellg());
            is.seekg(0);
            is.read(&buf[0], buf.size());
            if (!is) throw BadFile(path);
            image = decode_raw(&buf[0], buf.size());
        }
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

    void encode_raw (cv::Mat m, string *s) {
        std::ostringstream ss;
        int type = m.type();
        int rows = m.rows;
        int cols = m.cols;
        int elemSize = m.elemSize();
        ss.write(reinterpret_cast<char const *>(&type), sizeof(type));
        ss.write(reinterpret_cast<char const *>(&rows), sizeof(rows));
        ss.write(reinterpret_cast<char const *>(&cols), sizeof(cols));
        ss.write(reinterpret_cast<char const *>(&elemSize), sizeof(elemSize));
        for (int i = 0; i < rows; ++i) {
            ss.write(m.ptr<char const>(i), cols * elemSize);
        }
        *s = ss.str();
    }

    cv::Mat decode_raw (char const *buf, size_t sz) {
        if (sz < sizeof(int) * 4) return cv::Mat();
        int type = *reinterpret_cast<int const *>(buf); buf += sizeof(int); sz -= sizeof(int);
        int rows = *reinterpret_cast<int const *>(buf); buf += sizeof(int); sz -= sizeof(int);
        int cols = *reinterpret_cast<int const *>(buf); buf += sizeof(int); sz -= sizeof(int);
        int elemSize = *reinterpret_cast<int const *>(buf); buf += sizeof(int); sz -= sizeof(int);
        if (sz != rows * cols * elemSize) return cv::Mat();
        cv::Mat m(rows, cols, type);
        if (m.elemSize() != elemSize) return cv::Mat();
        size_t line = cols * elemSize;
        for (int i = 0; i < rows; ++i) {
            std::copy(buf, buf + line, m.ptr<char>(i));
            buf += line;
        }
        return m;
    }

    cv::Mat decode_buffer (const_buffer imbuf, int mode) {
        cv::Mat image = cv::imdecode(cv::Mat(1, boost::asio::buffer_size(imbuf), CV_8U,
                        const_cast<void *>(boost::asio::buffer_cast<void const *>(imbuf))),
                        mode);
        if (!image.data) {
            image = decode_raw(boost::asio::buffer_cast<char const *>(imbuf),
                        boost::asio::buffer_size(imbuf));
        }
        return image;
    }


}

