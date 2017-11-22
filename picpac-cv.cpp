#include <iostream>
#include <sstream>
#include <json11.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/fstream.hpp>
#include "picpac-cv.h"

namespace picpac {

    vector<cv::Scalar> PALETTE_TABLEAU20{
        {0, 0, 0},
		{31, 119, 180}, {174, 199, 232}, {255, 127, 14}, {255, 187, 120},
        {44, 160, 44}, {152, 223, 138}, {214, 39, 40}, {255, 152, 150},
        {148, 103, 189}, {197, 176, 213}, {140, 86, 75}, {196, 156, 148},
        {227, 119, 194}, {247, 182, 210}, {127, 127, 127}, {199, 199, 199},
        {188, 189, 34}, {219, 219, 141}, {23, 190, 207}, {158, 218, 229}
    };

    vector<cv::Scalar> PALETTE_TABLEAU20A{
        {0, 0, 0},
        {188, 189, 34}, {219, 219, 141}, {23, 190, 207}, {158, 218, 229},
        {44, 160, 44}, {152, 223, 138}, {214, 39, 40}, {255, 152, 150},
		{31, 119, 180}, {174, 199, 232}, {255, 127, 14}, {255, 187, 120},
        {227, 119, 194}, {247, 182, 210}, {127, 127, 127}, {199, 199, 199},
        {148, 103, 189}, {197, 176, 213}, {140, 86, 75}, {196, 156, 148}
    };

    using namespace json11;

    float LimitSize (cv::Mat input, int min_size, int max_size, cv::Mat *output) {
        if (input.rows == 0) {
            *output = cv::Mat();
            return 0;
        }
        float scale = 1.0;
        cv::Size sz(input.cols, input.rows);
        int maxs = std::max(sz.width, sz.height);

        if ((max_size > 0) && (maxs > max_size)) {
            scale = 1.0 * max_size / maxs;
            sz = cv::Size(sz.width * max_size / maxs, sz.height * max_size / maxs);
        }
        // large side > max
        int mins = std::min(sz.width, sz.height);
        if ((min_size > 0) && (mins < min_size)) {
            scale *= 1.0 * min_size / mins;
            sz = cv::Size(sz.width * min_size / mins, sz.height * min_size / mins);
        }
        if ((sz.width != input.cols) || (sz.height != input.rows)) {
            cv::Mat tmp;
            cv::resize(input, tmp, sz);
            input = tmp;
        }
        *output = input;
        return scale;
    }

    float LimitSizeBelow (cv::Mat input, int max_size, cv::Mat *output) {
        if (input.rows == 0) {
            *output = cv::Mat();
            return 0;
        }
        float scale = 1.0;
        int maxs = std::min(input.cols, input.rows);

        if ((max_size > 0) && (maxs > max_size)) {
            cv::Mat tmp;
            scale = 1.0 * maxs / max_size;
            cv::resize(input, tmp, cv::Size(input.cols * max_size / maxs, input.rows * max_size / maxs));
            input = tmp;
        }
        *output = input;
        return scale;
    }

    void check_add_label (Shape const *shape, Json::object *obj) {
        if (!shape->haveLabel()) return;
        cv::Scalar c = shape->label();
        Json::array v{c[0], c[1], c[2]};
        (*obj)["label"] = v;
    }

    class Box: public Shape {
    protected:
        cv::Rect_<float> rect;
        Box (char const *t): Shape(t) {}
    public:
        Box (Json const &geo, char const *t = "rect"): Shape(t) {
            rect.x = geo["x"].number_value();
            rect.y = geo["y"].number_value();
            rect.width = geo["width"].number_value();
            rect.height = geo["height"].number_value();
        }

        virtual void points (cv::Size sz, vector<cv::Point2f> *pts) const {
            float x = sz.width * rect.x;
            float y = sz.height * rect.y;
            float width = sz.width * rect.width-1;
            float height = sz.height * rect.height-1;
            pts->emplace_back(x, y);
            pts->emplace_back(x+width, y);
            pts->emplace_back(x+width, y+height);
            pts->emplace_back(x, y+height);
            return 4;
        }

        virtual void dump (Json *json) const {
            Json::object obj{
                {"type", type()},
                {"geometry", Json::object{
                                      {"x", rect.x},
                                      {"y", rect.y},
                                      {"width", rect.width},
                                      {"height", rect.height}
                                   }}
            };
            check_add_label(this, &obj);
            *json = obj;
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

#if 0
    class Ellipse: public Box {
    protected:
        Ellipse (char const *t): Box(t) {}
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

    class Point: public Ellipse {
    public:
        Point (Json const &geo, cv::Size size, ImageLoader::Config const &config): Ellipse("point") { 
            float x = geo["x"].number_value();
            float y = geo["y"].number_value();
            float xr = 1.0 * config.point_radius / size.width;
            float yr = 1.0 * config.point_radius / size.height;
            rect.x = x - xr;
            rect.y = y - yr;
            rect.width = 2 * xr;
            rect.height = 2 * yr;
        }
        virtual std::shared_ptr<Shape> clone () const {
            return std::shared_ptr<Shape>(new Point(*this));
        }
        virtual void draw (cv::Mat *m, cv::Scalar v, int thickness) const {
            if (rect.width == 0 && rect.height == 0) {
                // this is really a point
                int x = int(round(m->cols * rect.x));
                int y = int(round(m->rows * rect.y));
                // we really want a single point
                // and have to rely on that cv::line doesn't draw the end point
                cv::line(*m, cv::Point(x,y), cv::Point(x+1, y), v);
            }
            else {
                Ellipse::draw(m, v, thickness);
            }
        }
    };

#endif
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
            Json::object obj{
                {"type", type()},
                {"geometry", Json::object{{"points", std::move(pts)}}}
            };
            check_add_label(this, &obj);
            *json = obj;
        }
        virtual std::shared_ptr<Shape> clone () const {
            return std::shared_ptr<Poly>(new Poly(*this));
        }

        virtual void points (cv::Size sz, vector<cv::Point2f> *pts) const {
            for (unsigned i = 0; i < ps.size(); ++i) {
                auto const &from = points[i];
                pts->emplace_back(from.x * sz.width, from.y * sz.height);
            }
            return ps.size() 
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

    std::shared_ptr<Shape> Shape::create (Json const &geo, cv::Size size, ImageLoader::Config const &config) {
        string type = geo["type"].string_value();
        std::shared_ptr<Shape> shape;
        if (type == "rect") {
            shape = std::shared_ptr<Shape>(new Box(geo["geometry"]));
        }
#if 0
        else if (type == "ellipse") {
            shape = std::shared_ptr<Shape>(new Ellipse(geo["geometry"]));
        }
        else if (type == "polygon") {
            shape = std::shared_ptr<Shape>(new Poly(geo["geometry"]));
        }
        else if (type == "point") {
            shape = std::shared_ptr<Shape>(new Point(geo["geometry"], size, config));
        }
#endif
        else {
            CHECK(0) << "unknown shape: " << type;
        }
        Json const &label = geo["label"];
        if (label.is_null()) {
            ;
        }
        else if (label.is_array()) {
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
        return shape;
    }

    Annotation::Annotation (string const &txt, cv::Size size, ImageLoader::Config const &config) {
        string err;
        Json json = Json::parse(txt, err);
        if (err.size()) {
            LOG(ERROR) << "Bad json: " << err << " (" << txt << ")";
            return;
        }
        for (auto const &x: json["shapes"].array_items()) {
            shapes.emplace_back(Shape::create(x, size, config));
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

    void Annotation::points (cv::Size sz, AnnoPoints *anno) const {
        for (auto const &p: shapes) {
            int c = p->points(sz, &anno->points);
            anno->sz.push_bakc(c);
            float l = 1.0;
            if (p->haveLabel()) {
                l = p->label()[0];
            }
            anno->labels.push_back(l);
        }
    }

    void Annotation::draw (cv::Mat *m, cv::Scalar v, int thickness, vector<cv::Scalar> const *palette, bool show_number) const {
        static constexpr int font_face = cv::FONT_HERSHEY_SIMPLEX;
		static constexpr double font_scale = 0.4;
        
        int cc = 0;
        for (auto const &p: shapes) {
            cv::Scalar vv = v;
            if (p->haveLabel()) {
                vv = p->label();
            }
            if (palette) {
                unsigned idx = unsigned(vv[0]);
                if ((vv[1] != 0) || (vv[2] != 0) || idx >= palette->size()) {
                    idx = 0;
                }
                vv = palette->at(idx);
            }
            p->draw(m, vv, thickness);
            if (show_number) {
                cv::Rect_<float> bbox;
                p->bbox(&bbox);
                int px = bbox.br().x * m->cols;
                int py = bbox.br().y * m->rows;
                //std::cerr << px << ' ' << py << "   ++++++" << std::endl;
                cv::putText(*m, boost::lexical_cast<string>(cc), cv::Point(px, py), font_face, font_scale, vv);
            }
            ++cc;
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

    cv::Mat ImageLoader::preload_image (const_buffer buffer, LoadState *state) const {
        cv::Mat image = decode_buffer(buffer, config.decode_mode);
        if ((config.channels > 0) && config.channels != image.channels()) {
            cv::Mat tmp;
            if (image.channels() == 3 && config.channels == 1) {
                cv::cvtColor(image, tmp, CV_BGR2GRAY);
            }
            else if (image.channels() == 4 && config.channels == 1) {
                cv::cvtColor(image, tmp, CV_BGRA2GRAY);
            }
            else if (image.channels() == 4 && config.channels == 3) {
                cv::cvtColor(image, tmp, CV_BGRA2BGR);
            }
            else if (image.channels() == 1 && config.channels == 3) {
                cv::cvtColor(image, tmp, CV_GRAY2BGR);
            }
#ifdef SUPPORT_AUDIO_SPECTROGRAM
            else if (image.type() == CV_32FC2 && config.channels == 1) {
                spectrogram_to_gray(image, &tmp);
            }
            else if (image.type() == CV_32FC2 && config.channels == 3) {
                spectrogram_to_bgr(image, &tmp);
            }
#endif
            else CHECK(0) << "channel format not supported: from "
                          << image.channels()
                          << " to " << config.channels;
            image = tmp;
        }
        if (config.resize_width > 0 && config.resize_height > 0) {
            cv::resize(image, image, cv::Size(config.resize_width, config.resize_height), 0, 0);
        }
        else if (config.max_size > 0 || config.min_size > 0) {
            cv::Mat tmp;
            LimitSize(image, config.min_size, config.max_size, &tmp);
            image = tmp;
        }
        state->size = image.size();
        if (annotate == ANNOTATE_JSON && config.anno_copy && !state->copy_for_anno.data) {
            state->copy_for_anno = image.clone();
        }
        return image;
    }

    void ImageLoader::preload_annotation (const_buffer buffer, LoadState *state, AnnoPoints *anno) const {
        if (annotate == ANNOTATE_IMAGE) {
            CHECK(0);
        }
        else if (annotate == ANNOTATE_JSON) {
            if (config.anno_copy) {
                CHECK(0);
            }
            if (boost::asio::buffer_size(buffer) > 1) {
                char const *ptr = boost::asio::buffer_cast<char const *>(buffer);
                char const *ptr_end = ptr + boost::asio::buffer_size(buffer);
                Annotation a(string(ptr, ptr_end), state->size, config);

                a.points(state->size, anno);

                do {
                    if (!(config.anno_min_ratio > 0)) break;
                    CHECK(0);
                } while (false);
            }
        }
    }


    cv::Mat ImageLoader::process_image (cv::Mat image, PerturbVector const &p, LoadState const *state, bool is_anno) const {
        //TODO: scale might break min and/or max restriction
        auto CV_INTER = is_anno ? cv::INTER_NEAREST : cv::INTER_LINEAR;
        if (state->crop) {
            image = image(state->crop_bb);
            //cv::resize(im, image, image.size(), 0, 0, CV_INTER);
        }

        if (config.perturb) {
            if (p.angle != 0) {
                cv::Mat rot = cv::getRotationMatrix2D(cv::Point(image.cols/2, image.rows/2), p.angle, p.scale);
                {
                    cv::Mat tmp;
                    cv::warpAffine(image, tmp, rot, image.size(), CV_INTER, config.pert_border);
                    //cv::resize(tmp, tmp, cv::Size(), p.scale, p.scale);
                    image = tmp;
                }
            }
            else if (p.scale != 1) {
                {
                    cv::Mat tmp;
                    cv::resize(image, tmp, cv::Size(), p.scale, p.scale, CV_INTER);
                    image = tmp;
                }
            }
            /*
            std::cout << p.color[0] << " " << p.color[1] << " " << p.color[2] << std::endl;
            std::cout << colorspace << std::endl;
            */
            if (!is_anno) {
                cv::Scalar pert_color = p.color;
                if (image.channels() == 3) {
                    if (image.type() == CV_16UC3) {
                        image.convertTo(image, CV_32FC3);
                    }
                    if (colorspace == COLOR_Lab) {
                        cv::cvtColor(image, image, CV_BGR2Lab);
                    }
                    else if (colorspace  == COLOR_HSV) {
                        cv::cvtColor(image, image, CV_BGR2HSV);
                    }
                    else if (colorspace == COLOR_SAME) {
                        pert_color[1] = pert_color[2] = pert_color[0];
                    }
                }
                image += pert_color;
                if (image.channels() == 3) {
                    if (colorspace == COLOR_Lab) {
                        cv::cvtColor(image, image, CV_Lab2BGR);
                    }
                    else if (colorspace  == COLOR_HSV) {
                        cv::cvtColor(image, image, CV_HSV2BGR);
                    }
                }
            }

            if (p.hflip && p.vflip) {
                cv::flip(image, image, -1);
            }
            else if (p.hflip && !p.vflip) {
                cv::flip(image, image, 1);
            }
            else if (!p.hflip && p.vflip) {
                cv::flip(image, image, 0);
            }
        }
        if ((config.crop_width > 0) && (config.crop_height > 0)
            && ((image.cols > config.crop_width)
            || (image.rows > config.crop_height))) {
            // cropping
            int marginx = image.cols - config.crop_width;
            int marginy = image.rows - config.crop_height;
            if (!config.perturb) {

                marginx /=2;
                marginy /=2;
            }
            else {
                marginx = p.shiftx % (marginx + 1);
                marginy = p.shifty % (marginy + 1);
            }

            cv::Rect roi(marginx,
                         marginy,
                         config.crop_width,
                         config.crop_height);
            image = image(roi);
        }
        if (config.round_div > 0) {
            int width = image.cols;
            int height = image.rows;
            width = width / config.round_div * config.round_div + config.round_mod;
            height = height / config.round_div * config.round_div + config.round_mod;
            if (width > image.cols) {
                width -= config.round_div;
            }
            if (height > image.rows) {
                height -= config.round_div;
            }
            CHECK((width > 0) && (height > 0));
            int marginx = image.cols - width;
            int marginy = image.rows - height;
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
            image = image(roi);
        }
        return image;
    }

    void ImageLoader::process_annotation (AnnoPoints *anno, PerturbVector const &p, LoadState const *state) const {
        //TODO: scale might break min and/or max restriction
        vector<cv::Point2f > *pts = &anno->points;
        cv::Size sz = anno->size;
        if (state->crop) {
            CHECK(0);
            for (auto &p: *pts) {
                p.x -= state->crop_bb.x;
                p.y -= state->crop_bb.y;
            }
            sz = state->crop_bb.size();
        }

        if (config.perturb) {
            if (p.angle != 0) {
                cv::Mat rot = cv::getRotationMatrix2D(cv::Point(sz.width/2, sz.height/2), p.angle, p.scale);
                cv::transform(*pts, *pts, rot);
            }
            else if (p.scale != 1) {
                cv::Size newsz(round(p.scale * sz.width),
                               round(p.scale * sz.height));
                float xs = 1.0 * newsz.width / sz.width;
                float ys = 1.0 * newsz.height / sz.height;
                std::cout << p.scale << " " << xs << " " << ys << std::endl;
                for (auto &p: *pts) {
                    p.x *= xs;
                    p.y *= ys;
                }
                sz = newsz;
            }
            /*
            std::cout << p.color[0] << " " << p.color[1] << " " << p.color[2] << std::endl;
            std::cout << colorspace << std::endl;
            */
            if (p.hflip && p.vflip) {
                for (auto &p: *pts) {
                    p.x = sz.width - p.x;
                    p.y = sz.height - p.y;
                }
            }
            else if (p.hflip && !p.vflip) {
                for (auto &p: *pts) {
                    p.x = sz.width - p.x;
                }
            }
            else if (!p.hflip && p.vflip) {
                for (auto &p: *pts) {
                    p.y = sz.height - p.y;
                }
            }
        }
        if ((config.crop_width > 0) && (config.crop_height > 0)) {
            CHECK(0);
        }

        if (config.round_div > 0) {
            int width = sz.width;
            int height = sz.height;
            width = width / config.round_div * config.round_div + config.round_mod;
            height = height / config.round_div * config.round_div + config.round_mod;
            if (width > sz.width) {
                width -= config.round_div;
            }
            if (height > sz.height) {
                height -= config.round_div;
            }
            CHECK((width > 0) && (height > 0));
            int marginx = sz.width - width;
            int marginy = sz.height - height;
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
            for (auto &p: *pts) {
                p.x -= roi.x;
                p.y -= roi.y;
            }
            sz = roi.size();
        }
        anno->size = sz;
    }

    cv::Mat ImageLoader::preload_annotation_map (const_buffer buffer, LoadState *state) const {
        cv::Mat annotation;
        if (annotate == ANNOTATE_IMAGE) {
            CHECK(0);
        }
        else if (annotate == ANNOTATE_JSON) {
            cv::Mat anno;
            if (config.anno_copy) {
                CHECK(0);
            }
            else {
                anno = cv::Mat(state->size, config.anno_type, cv::Scalar(0));
            }
            if (boost::asio::buffer_size(buffer) > 1) {
                char const *ptr = boost::asio::buffer_cast<char const *>(buffer);
                char const *ptr_end = ptr + boost::asio::buffer_size(buffer);
                Annotation a(string(ptr, ptr_end), state->size, config);
                a.number_shapes();
                cv::Scalar color(config.anno_color1,
                             config.anno_color2,
                             config.anno_color3);
                auto const *palette = &PALETTE_TABLEAU20;
                if (anno_palette == ANNOTATE_PALETTE_NONE) {
                    palette = nullptr;
                }
                else if (anno_palette == ANNOTATE_PALETTE_TABLEAU20A) {
                    palette = &PALETTE_TABLEAU20A;
                }
                
                a.draw(&anno, color, config.anno_thickness, palette, config.anno_number);

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
                    state->crop = true;
                    state->crop_bb = bbox;
                } while (false);
            }
            annotation = anno;
        }
        return annotation;
    }

#if 0
    cv::Mat ImageLoader::load_image (const_buffer buffer, PerturbVector const &pv, LoadState *state) const {
        cv::Mat image = preload_image(buffer, state);
        return process_image(image, pv, state, false);
    }

    void ImageLoader::load_annotation (AnnoPoints *anno, const_buffer buffer, PerturbVector const &pv, LoadState *state) const {
        preload_annotation(buffer, state, anno);
        process_annotation(&anno->points, pv, state);
    }
#endif

    float static jaccard (cv::Rect_<float> r1,
                          cv::Rect_<float> r2) {
        cv::Rect_<float> sect = r1 & r2;
        float rs = sect.area();
        return rs / (r1.area() + r2.area() - rs);
    }

    struct TruthBox {
        float value;
        cv::Rect_<float> box;

        float score;
        float *mask;
        float *shifts;
        float *label;
        float sh[4];
    };

    void setup_dirs (vector<float> *dirs,
                     vector<float> *dirs_mask,
                     cv::Mat const &p_map,
                     vector<vector<cv::Point_<float>>> const &polys,
                     float x, float y) {
    }

    void ImageLoader::setup_labels (cv::Mat image, 
                                    cv::Size sz,
                                    AnnoPoints const &anno,
                                    cv::Mat p_map,
                                    vector<float> *labels,
                                    vector<float> *mask,
                                    vector<float> *shifts,
                                    vector<float> *dirs_mask,
                                    vector<float> *dirs,
                                    int *cnt) const {
        vector<TruthBox> truths;
        for (auto &p: anno.points) {
            p.x /= config.downsize;
            p.y /= config.downsize;
        }

        for (unsigned i = 0, i2 = 0; i2 < anno.boxes; i += 4, i2 += 1) {
            CHECK(anno.sz[i2] == 4);
            CHECK(i + 4 <= anno.points.size());
            float minx = anno.points[i].x;
            float maxx = minx;
            float miny = anno.points[i].y;
            float maxy = miny;
            for (unsigned j = 1; j < 4; ++j) {
                minx = std::min(anno.points[i+j].x, minx);
                maxx = std::max(anno.points[i+j].x, maxx);
                miny = std::min(anno.points[i+j].y, miny);
                maxy = std::max(anno.points[i+j].y, maxy);
            }
            TruthBox b;
            b.value = anno.labels[i2];
            b.box = cv::Rect_<float>(minx, miny, maxx-minx, maxy-miny);
            b.score = 0;
            b.mask = nullptr;
            b.shifts = nullptr;
            b.label = nullptr;
            truths.push_back(b);
        }
        vector<vector<Point_<float>>> polys;
        unsigned off = anno.boxes * 4;
        for (int i = anno.boxes; i < anno.labels; ++i) {
            int cc = anno.sz[i];
            // polygon has cc points
            polys.emplace_back();
            if (cc == 0) continue;
            auto &v = polys.back();
            unsigned off0 = off;
            for (int j = 0; j < cc; ++j) {
                v.push_back(anno.points[off++])
            }
            v.push_back(anno.points[off0]);
        }
        CHECK(off == anno.points.size());
#if 0
        std::cout << image.rows << 'x' << image.cols << " => " << sz.height << 'x' << sz.width << std::endl;
        static int serial = 0;
        cv::Mat out;
        cv::resize(image, out, sz);
        for (auto const &r: truths) {
            cv::Rect ri(int(r.box.x), int(r.box.y), int(r.box.width), int(r.box.height));
            cv::rectangle(out, ri, cv::Scalar(0, 0xff, 0));
        }
        cv::imwrite("test/" + lexical_cast<string>(serial++) + ".jpg", out);
        if (serial >= 25) exit(0);
#endif

        // boxes
        vector<cv::Size_<float>> dsizes(config.boxes);
        for (auto &b: dsizes) {
            b.width /= config.downsize;
            b.height /= config.downsize;
        }

        float *pm = &(*mask)[0];
        float *ps = &(*shifts)[0];
        float *pl = &(*labels)[0];
        *cnt = 0;
        //std::cerr << sz.height << 'x' << sz.width << std::endl;
        for (int y = 0; y < sz.height; ++y) {
            for (int x = 0; x < sz.width; ++x) {
                // determine direction
                // lookup p_map
                // 
                for (auto const &sz: dsizes) {
                    // for each default box
                    cv::Rect_<float> dbox(x - sz.width/2,
                                         y - sz.height/2,
                                         sz.width,
                                         sz.height);
                    float best = config.ssd_th;
                    bool used = false;
                    for (auto &truth: truths) {
                        float score = jaccard(truth.box, dbox);
                        if (score > truth.score) {
                            truth.score = score;
                            truth.mask = pm;
                            truth.shifts = ps;
                            truth.label = pl;
                            truth.sh[0] = truth.box.x + truth.box.width/2 - x;
                            truth.sh[1] = truth.box.y + truth.box.height/2 - y;
                            truth.sh[2] = truth.box.width - dbox.width;
                            truth.sh[3] = truth.box.height - dbox.height;
                            // setup dirs
                            setup_dirs(dirs, dirs_mask,
                                       p_map, polys,
                                       // x, y
                                       truth.box.x + truth.box.width / 2,
                                       truth.box.y + truth.box.height);
                        }
                        if (score > best) {
                            best = score;
                            used = true;
                            pl[0] = truth.value;
                            pm[0] = 1.0;
                            pm[1] = 1.0;
                            pm[2] = 1.0;
                            pm[3] = 1.0;
                            ps[0] = truth.box.x + truth.box.width/2 - x;
                            ps[1] = truth.box.y + truth.box.height/2 - y;
                            ps[2] = truth.box.width - dbox.width;
                            ps[3] = truth.box.height - dbox.height;
                        }
                    }
                    if (used) {
                        ++*cnt;
                    }
                    ++pl;
                    pm += 4;
                    ps += 4;
                }
            }
        }
        for (auto &truth: truths) {
            /*
            if (truth.label == 0) {
                std::cerr << "MISS: " << truth.box.x << ',' << truth.box.y << ' ' << truth.box.height << 'x' << truth.box.width << std::endl;
            }
            */
            /*
            std::cout << truth.score << std::endl;
            CHECK(truth.label);
            */
            if (truth.label == nullptr) continue;   // not found
            if (truth.label[0] == 0) {
                ++*cnt;
            }
            // otherwise still set 
            truth.label[0] = truth.value;
            truth.mask[0] = 1;
            truth.mask[1] = 1;
            truth.mask[2] = 1;
            truth.mask[3] = 1;
            truth.shifts[0] = truth.sh[0];
            truth.shifts[1] = truth.sh[1];
            truth.shifts[2] = truth.sh[2];
            truth.shifts[3] = truth.sh[3];
            setup_dirs(dirs, dirs_mask,
                       p_map, polys,
                       // x, y
                       truth.box.x + truth.box.width / 2,
                       truth.box.y + truth.box.height);
        }
    }

    void ImageLoader::load (RecordReader rr, PerturbVector const &pv, Value *value,
           CacheValue *cache, std::mutex *mutex) const {
        CHECK(cache == nullptr);
        CHECK(mutex == nullptr);
        Record r;
        rr(&r); // disk access
        value->label = r.meta().label;
        LoadState state;
        cv::Mat image = preload_image(r.field(0), &state);
        value->image = process_image(image, pv, &state, false);

        CHECK(value->image.rows % config.downsize == 0);
        CHECK(value->image.cols % config.downsize == 0);
        cv::Size lsize(value->image.cols/config.downsize,
                        value->image.rows/config.downsize);
        vector<float> labels(lsize.width * lsize.height * config.boxes.size(), 0);
        vector<float> mask(labels.size() * 4, 0);
        vector<float> shifts(mask.size(), 0);
        vector<float> dirs_mask(lsize.width * lsize.height * 2, 0);
        vector<float> dirs(dirs_mask.size(), 0);

        int matched = 0;
        if (annotate != ANNOTATE_NONE && r.meta().width > 1) {
            AnnoPoints anno;
            anno.size = image.size();
            preload_annotation(r.field(1), &state, &anno);
            process_annotation(&anno, pv, &state);
            anno.boxes = anno.labels.size();
            cv::Mat p_map(lsize, config.anno_type, cv::Scalar(0));
            if (r.meta().width > 2) {
                preload_annotation(r.field(2), &state, &anno);
                cv::Mat map = preload_annotation_map(r.field(2), &state);
                map = process_image(map, pv, &state, true);
                cv::resize(map, p_map, lsize, cv::INTER_NEAREST);
                process_annotation((&anno, pv, &state);
            }
            setup_labels(value->image, lsize, anno, p_map, &labels, &mask, &shifts, &dirs_mask, &dirs, &matched);
        }
        value->label = matched > 0 ? 1 : 0;
        value->matched_boxes = matched;
        value->label_size = lsize;
        value->labels = labels;
        value->mask.swap(mask);
        value->shift.swap(shifts);
        value->dirs_mask.swap(dirs_mask);
        value->dirs.swap(dirs);
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
        cv::imencode(code.empty() ? ".jpg": code, image, buffer, _params);
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
            LimitSizeBelow(image, max, &rs);
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

    void ImageReader::transcode (string const &binary, string *data) {
        bool do_code = code.size() || (mode != cv::IMREAD_UNCHANGED);
        cv::Mat buffer(1, binary.size(), CV_8U, const_cast<void *>(reinterpret_cast<void const *>(&binary[0])));
        cv::Mat image = cv::imdecode(buffer, mode);
        if (!image.data) { // try raw
            image = decode_raw(&binary[0], binary.size());
        }
        if (!image.data) throw BadFile("");
        if (resize > 0) {
            cv::resize(image, image, cv::Size(resize, resize));
            do_code = true;
        }
        else if (max > 0) {
            cv::Mat rs;
            LimitSizeBelow(image, max, &rs);
            if (rs.total() != image.total()) {
                image = rs;
                do_code = true;
            }
        }
        if (do_code) {
            encode(image, data);
        }
        else {
            *data = binary;
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

