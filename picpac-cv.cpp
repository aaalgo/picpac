#include <iostream>
#include <sstream>
#include <json11.hpp>
#include <boost/lexical_cast.hpp>
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
            if (thickness == cv::FILLED) {
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
        else if (type == "ellipse") {
            shape = std::shared_ptr<Shape>(new Ellipse(geo["geometry"]));
        }
        else if (type == "polygon") {
            shape = std::shared_ptr<Shape>(new Poly(geo["geometry"]));
        }
        else if (type == "point") {
            shape = std::shared_ptr<Shape>(new Point(geo["geometry"], size, config));
        }
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
        cv::cvtColor(o, *bgr, cv::COLOR_HSV2BGR);
    }

    cv::Mat ImageLoader::preload_image (const_buffer buffer, LoadState *state) const {
        cv::Mat image = decode_buffer(buffer, config.decode_mode);
        if ((config.channels > 0) && config.channels != image.channels()) {
            cv::Mat tmp;
            if (image.channels() == 3 && config.channels == 1) {
                cv::cvtColor(image, tmp, cv::COLOR_BGR2GRAY);
            }
            else if (image.channels() == 4 && config.channels == 1) {
                cv::cvtColor(image, tmp, cv::COLOR_BGRA2GRAY);
            }
            else if (image.channels() == 4 && config.channels == 3) {
                cv::cvtColor(image, tmp, cv::COLOR_BGRA2BGR);
            }
            else if (image.channels() == 1 && config.channels == 3) {
                cv::cvtColor(image, tmp, cv::COLOR_GRAY2BGR);
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

    cv::Mat ImageLoader::preload_annotation (const_buffer buffer, LoadState *state) const {
        cv::Mat annotation;
        if (annotate == ANNOTATE_IMAGE) {
            if (boost::asio::buffer_size(buffer) == 0) {
                annotation = cv::Mat(state->size, CV_8U, cv::Scalar(0));
            }
            else {
                annotation = cv::imdecode(cv::Mat(1, boost::asio::buffer_size(buffer), CV_8U,
                            const_cast<void *>(boost::asio::buffer_cast<void const *>(buffer))),
                            cv::IMREAD_UNCHANGED);
                if (!annotation.data) {
                    annotation = decode_raw(boost::asio::buffer_cast<char const *>(buffer),
                                        boost::asio::buffer_size(buffer));
                }
                auto const *palette = &PALETTE_TABLEAU20;
                if (anno_palette == ANNOTATE_PALETTE_NONE) {
                    palette = nullptr;
                }
                else if (anno_palette == ANNOTATE_PALETTE_TABLEAU20A) {
                    palette = &PALETTE_TABLEAU20A;
                }
                if (annotation.data && palette) {
                    CHECK(annotation.type() == CV_8UC1);
                    cv::Mat mat(annotation.rows,
                                annotation.cols, CV_8UC3);
                    // apply pallet
                    for (int i = 0; i < mat.rows; ++i) {
                        uint8_t const *from = annotation.ptr<uint8_t const>(i);
                        uint8_t *to = mat.ptr<uint8_t>(i);
                        for (int j = 0; j < mat.cols; ++j) {
                            unsigned x = from[j];
                            if (x >= palette->size()) x = 0;
                            auto c = palette->at(x);
                            to[0] = c[0];
                            to[1] = c[1];
                            to[2] = c[2];
                            to += 3;
                        }
                    }
                    annotation = mat;
                }
                if (annotation.size() != state->size) {
                    cv::resize(annotation, annotation, state->size, 0, 0, cv::INTER_NEAREST);
                }
            }
            CHECK(config.anno_min_ratio == 0) << "Not supported";
        }
        else if (annotate == ANNOTATE_JSON) {
            cv::Mat anno;
            if (config.anno_copy) {
                anno = state->copy_for_anno;
                CHECK(anno.data);
                state->copy_for_anno = cv::Mat();
            }
            else {
                anno = cv::Mat(state->size, config.anno_type, cv::Scalar(0));
            }
            if (boost::asio::buffer_size(buffer) > 1) {
                char const *ptr = boost::asio::buffer_cast<char const *>(buffer);
                char const *ptr_end = ptr + boost::asio::buffer_size(buffer);
                Annotation a(string(ptr, ptr_end), state->size, config);
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


    cv::Mat ImageLoader::process_image (cv::Mat image, PerturbVector const &p, LoadState const *state, bool is_anno) const {
        //TODO: scale might break min and/or max restriction
        auto CV_INTER = is_anno ? cv::INTER_NEAREST : cv::INTER_LINEAR;
        if (state->crop) {
            image = image(state->crop_bb);
            //cv::resize(im, image, image.size(), 0, 0, CV_INTER);
        }

        if (config.perturb) {
            if (p.angle != 0) {
                cv::Mat rot = cv::getRotationMatrix2D(cv::Point(image.cols/2, image.rows/2), p.angle, 1.0);
                {
                    cv::Mat tmp;
                    cv::warpAffine(image, tmp, rot, image.size(), CV_INTER, config.pert_border);
                    //cv::resize(tmp, tmp, cv::Size(), p.scale, p.scale);
                    image = tmp;
                }
            }
            if (p.scale != 1) {
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
                        cv::cvtColor(image, image, cv::COLOR_BGR2Lab);
                    }
                    else if (colorspace  == COLOR_HSV) {
                        cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
                    }
                    else if (colorspace == COLOR_SAME) {
                        pert_color[1] = pert_color[2] = pert_color[0];
                    }
                }
                image += pert_color;
                if (image.channels() == 3) {
                    if (colorspace == COLOR_Lab) {
                        cv::cvtColor(image, image, cv::COLOR_Lab2BGR);
                    }
                    else if (colorspace  == COLOR_HSV) {
                        cv::cvtColor(image, image, cv::COLOR_HSV2BGR);
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

    cv::Mat ImageLoader::load_image (const_buffer buffer, PerturbVector const &pv, LoadState *state) const {
        cv::Mat image = preload_image(buffer, state);
        return process_image(image, pv, state, false);
    }

    cv::Mat ImageLoader::load_annotation (const_buffer buffer, PerturbVector const &pv, LoadState *state) const {
        cv::Mat image = preload_annotation(buffer, state);
        return process_image(image, pv, state, true);
    }

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

    static cv::Mat crop_pad (cv::Mat image, cv::Size size, bool perturb, int shiftx, int shifty) {
        int from_x = 0, from_width = image.cols;
        int from_y = 0, from_height = image.rows;
        int to_x = 0, to_width = size.width;
        int to_y = 0, to_height = size.height;
        adjust_crop_pad_range(from_x, from_width, to_x, to_width, perturb, shiftx);
        adjust_crop_pad_range(from_y, from_height, to_y, to_height, perturb, shifty);
        cv::Mat to(size, image.type(), cv::Scalar(0,0,0));
        image(cv::Rect(from_x, from_y, from_width, from_height)).copyTo(to(cv::Rect(to_x, to_y, to_width, to_height)));
        return to;
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
                    cv::cvtColor(cached.image, tmp, cv::COLOR_BGR2GRAY);
                }
                else if (cached.image.channels() == 4 && config.channels == 1) {
                    cv::cvtColor(cached.image, tmp, cv::COLOR_BGRA2GRAY);
                }
                else if (cached.image.channels() == 4 && config.channels == 3) {
                    cv::cvtColor(cached.image, tmp, cv::COLOR_BGRA2BGR);
                }
                else if (cached.image.channels() == 1 && config.channels == 3) {
                    cv::cvtColor(cached.image, tmp, cv::COLOR_GRAY2BGR);
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
                    auto const *palette = &PALETTE_TABLEAU20;
                    if (anno_palette == ANNOTATE_PALETTE_NONE) {
                        palette = nullptr;
                    }
                    else if (anno_palette == ANNOTATE_PALETTE_TABLEAU20A) {
                        palette = &PALETTE_TABLEAU20A;
                    }
                    if (cached.annotation.data && palette) {
                        CHECK(cached.annotation.type() == CV_8UC1);
                        cv::Mat mat(cached.annotation.rows,
                                    cached.annotation.cols, CV_8UC3);
                        // apply pallet
                        for (int i = 0; i < mat.rows; ++i) {
                            uint8_t const *from = cached.annotation.ptr<uint8_t const>(i);
                            uint8_t *to = mat.ptr<uint8_t>(i);
                            for (int j = 0; j < mat.cols; ++j) {
                                unsigned x = from[j];
                                if (x >= palette->size()) x = 0;
                                auto c = palette->at(x);
                                to[0] = c[0];
                                to[1] = c[1];
                                to[2] = c[2];
                                to += 3;
                            }
                        }
                        cached.annotation = mat;
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
                    Annotation a(r.field_string(1), cached.image.size(), config);
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
                        cv::Mat im = cached.image(bbox);
                        cv::Mat an = anno(bbox);
                        cv::Mat rim, ran;
                        cv::resize(im, rim, cached.image.size());
                        cv::resize(an, ran, anno.size(), 0, 0, cv::INTER_NEAREST);
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
                /*
                CHECK(out->image.cols >= config.crop_width);
                CHECK(out->image.rows >= config.crop_height);
                // cropping
                int marginx = out->image.cols - config.crop_width;
                int marginy = out->image.rows - config.crop_height;
                cv::Rect roi(marginx / 2,
                             marginy / 2,
                             config.crop_width,
                             config.crop_height);
                */
                out->image = crop_pad(out->image, cv::Size(config.crop_width, config.crop_height), false , 0, 0);
                if (out->annotation.data) {
                    out->annotation = crop_pad(out->annotation, cv::Size(config.crop_width, config.crop_height), false , 0, 0);
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
            cv::Mat rot = cv::getRotationMatrix2D(cv::Point(image.cols/2, image.rows/2), p.angle, 1.0);
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
        if (p.scale != 1) {
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
        cv::Scalar pert_color = p.color;
        if (image.channels() == 3) {
            if (image.type() == CV_16UC3) {
                image.convertTo(image, CV_32FC3);
            }
            if (colorspace == COLOR_Lab) {
                cv::cvtColor(image, image, cv::COLOR_BGR2Lab);
            }
            else if (colorspace  == COLOR_HSV) {
                cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
            }
            else if (colorspace == COLOR_SAME) {
                pert_color[1] = pert_color[2] = pert_color[0];
            }
        }
        image += pert_color;
        if (image.channels() == 3) {
            if (colorspace == COLOR_Lab) {
                cv::cvtColor(image, image, cv::COLOR_Lab2BGR);
            }
            else if (colorspace  == COLOR_HSV) {
                cv::cvtColor(image, image, cv::COLOR_HSV2BGR);
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
        if ((config.crop_width > 0) && (config.crop_height > 0)) {
            // cropping
            out->image = crop_pad(out->image, cv::Size(config.crop_width, config.crop_height), true, p.shiftx, p.shifty);
            if (out->annotation.data) {
                out->annotation = crop_pad(out->annotation, cv::Size(config.crop_width, config.crop_height), true, p.shiftx, p.shifty);
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
        cv::imencode(code.empty() ? ".jpg": code, image, buffer, _params);
        char const *from = reinterpret_cast<char const *>(&buffer[0]);
        *data = string(from, from + buffer.size());
    }

    void ImageReader::read (fs::path const &path, string *data) {
        bool do_code = code.size() || (mode != cv::IMREAD_UNCHANGED);
        cv::Mat image = cv::imread(path.native(), mode);
        if (!image.data) { // try raw
            string buf;
            std::ifstream is(path, std::ios::binary);
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
            std::ifstream is(path, std::ios::binary);
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
        if (sz != size_t(rows * cols * elemSize)) return cv::Mat();
        cv::Mat m(rows, cols, type);
        if (int(m.elemSize()) != elemSize) return cv::Mat();
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

