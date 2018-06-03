#include "picpac-image.h"

namespace picpac {

    vector<cv::Scalar> PALETTE{
             {180, 119, 31}, {232, 199, 174}, {14, 127, 255}, {120, 187, 255},
			 {44, 160, 44}, {138, 223, 152}, {40, 39, 214}, {150, 152, 255},
			 {189, 103, 148}, {213, 176, 197}, {75, 86, 140}, {148, 156, 196},
			 {194, 119, 227}, {210, 182, 247}, {127, 127, 127}, {199, 199, 199},
			 {34, 189, 188}, {141, 219, 219}, {207, 190, 23}, {229, 218, 158}};

    cv::Scalar Shape::render_color (RenderOptions const &opt) const {
        if (opt.use_tag) {
            return cv::Scalar(tag, tag, tag);
        }
        if (opt.use_serial) {
            return cv::Scalar(serial, serial, serial);
        }
        if (opt.use_palette) {
            return PALETTE[rand() % PALETTE.size()];
        }
        else {
            return color;
        }
    }


    class Point: public Shape {
    public:
        Point (json const &geo, cv::Size sz): Shape("point") {
            float x = geo.at("x").get<float>() * sz.width;
            float y = geo.at("y").get<float>() * sz.height;
            controls.emplace_back(x, y);
        }
        virtual std::unique_ptr<Shape> clone () const {
            return std::unique_ptr<Shape>(new Point(*this));
        }
        virtual void render (cv::Mat *m, RenderOptions const &opt) const {
            cv::circle(*m, round(controls[0]), opt.point_radius, render_color(opt), opt.thickness, opt.line_type, opt.shift);
        }
    };


    class Rectangle: public Shape {
    public:
        Rectangle (json const &geo, cv::Size sz): Shape("box") {
            float x = geo.at("x").get<float>() * sz.width;
            float y = geo.at("y").get<float>() * sz.height;
            float w = geo.at("width").get<float>() * sz.width;
            float h = geo.at("height").get<float>() * sz.height;

            controls.emplace_back(x, y);
            controls.emplace_back(x+w, y+h);
            controls.emplace_back(x+w, y);
            controls.emplace_back(x, y + h);
        }

        virtual std::unique_ptr<Shape> clone () const {
            return std::unique_ptr<Shape>(new Rectangle(*this));
        }

        virtual void render (cv::Mat *m, RenderOptions const &opt) const {
            cv::rectangle(*m, round(controls[0]), round(controls[1]), render_color(opt), opt.thickness, opt.line_type, opt.shift);
        }

        virtual void transform (std::function<void(vector<cv::Point2f> *)> f) {
            // some shape might need pre-post processing
            float area = (controls[1].x - controls[0].x) * (controls[1].y - controls[0].y);
            f(&controls);
            float min_x = controls[0].x;
            float max_x = controls[0].x;
            float min_y = controls[0].y;
            float max_y = controls[0].y;
            for (int i = 1; i < 4; ++i) {
                min_x = std::min(min_x, controls[i].x);
                max_x = std::max(max_x, controls[i].x);
                min_y = std::min(min_y, controls[i].y);
                max_y = std::max(max_y, controls[i].y);
            }
            float mid_x = (min_x + max_x) / 2;
            float span_x = (max_x - min_x);
            float mid_y = (min_y + max_y) / 2;
            float span_y = (max_y - min_y);
            float rate = std::sqrt((area + 1.0) / (span_x * span_y + 1.0));
            float d_x = span_x * rate / 2;
            float d_y = span_y * rate / 2;

            min_x = mid_x - d_x;
            max_x = mid_x + d_x;
            min_y = mid_y - d_y;
            max_y = mid_y + d_y;


            controls[0].x = min_x;
            controls[0].y = min_y;
            controls[1].x = max_x;
            controls[1].y = max_y;
            controls[2].x = max_x;
            controls[2].y = min_y;
            controls[3].x = min_x;
            controls[3].y = max_y;
        }

#if 0
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
        virtual void bbox (cv::Rect_<float> *bb) const {
            *bb = rect;
        }
        /*
        virtual void zoom (cv::Rect_<float> const &bb) {
            rect.x -= bb.x;
            rect.y -= bb.y;
            rect.x /= bb.width;
            rect.width /= bb.width;
            rect.y /= bb.height;
            rect.height /= bb.height;
        }
        */

#endif

    };

    class Ellipse: public Rectangle {
    public:
        Ellipse (json const &geo, cv::Size sz): Rectangle(geo, sz) {
            type = "ellipse";
        }

        virtual std::unique_ptr<Shape> clone () const {
            return std::unique_ptr<Shape>(new Ellipse(*this));
        }
        virtual void render (cv::Mat *m, RenderOptions const &opt) const {
            CHECK(false) << "TODO";
            /*
            cv::Point2f center(m->cols * (rect.x + rect.width/2),
                               m->rows * (rect.y + rect.height/2));
            cv::Size2f size(m->cols * rect.width, m->rows * rect.height);
            cv::ellipse(*m, cv::RotatedRect(center, size, 0), v, thickness);
            */
        }
    };

    class Polygon: public Shape {
    public:
        Polygon (json const &geo, cv::Size sz): Shape("polygon") {
            for (auto const &p: geo.at("points")) {
                controls.emplace_back(p.at("x").get<float>() * sz.width, p.at("y").get<float>() * sz.height);
            }
        }

        virtual std::unique_ptr<Shape> clone () const {
            return std::unique_ptr<Shape>(new Polygon(*this));
        }

        virtual void render (cv::Mat *m, RenderOptions const &opt) const {
            vector<cv::Point> ps; 
            ps.reserve(controls.size());
            for (auto const &p: controls) {
                ps.push_back(round(p));
            }
            cv::Point const *pps = &ps[0];
            int const nps = ps.size();
            if (opt.thickness == CV_FILLED) {
                cv::fillPoly(*m, &pps, &nps, 1, render_color(opt), opt.line_type, opt.shift);
            }
            else {
                cv::polylines(*m, &pps, &nps, 1, true, render_color(opt), opt.thickness, opt.line_type, opt.shift);
            }
        }

#if 0
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

        virtual void draw (cv::Mat *m, cv::Scalar v, int thickness) const {
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
#endif
    };

    class Polygons: public Shape {
        vector<unsigned> sizes;
    public:
        Polygons (json const &geo, cv::Size sz): Shape("polygons") {
            for (auto const &polygon: geo.at("points")) {
                // array of array
                unsigned cc = 0;
                for (auto const &p: polygon) {
                    controls.emplace_back(p.at("x").get<float>() * sz.width, p.at("y").get<float>() * sz.height);
                    cc += 1;
                }
                sizes.push_back(cc);
            }
        }

        virtual std::unique_ptr<Shape> clone () const {
            return std::unique_ptr<Shape>(new Polygons(*this));
        }

        virtual void render (cv::Mat *m, RenderOptions const &opt) const {
            vector<cv::Point> ps; 
            ps.reserve(controls.size());
            unsigned off = 0;
            for (unsigned sz: sizes) {
                ps.clear();
                for (unsigned i = 0; i < sz; ++i, ++off) {
                    ps.push_back(round(controls[off]));
                }
                cv::Point const *pps = &ps[0];
                int const nps = ps.size();
                if (opt.thickness == CV_FILLED) {
                    cv::fillPoly(*m, &pps, &nps, 1, render_color(opt), opt.line_type, opt.shift);
                }
                else {
                    cv::polylines(*m, &pps, &nps, 1, true, render_color(opt), opt.thickness, opt.line_type, opt.shift);
                }
            }
        }
    };

    std::unique_ptr<Shape> Shape::create (json const &spec, cv::Size sz) {
        string type = spec.at("type").get<string>();
        auto geo = spec.at("geometry");

        std::unique_ptr<Shape> shape;
        if (type == "rect") {
            shape = std::unique_ptr<Shape>(new Rectangle(geo, sz));
        }
        else if (type == "polygon") {
            shape = std::unique_ptr<Shape>(new Polygon(geo, sz));
        }
        else if (type == "polygons") {
            shape = std::unique_ptr<Shape>(new Polygons(geo, sz));
        }
        else if (type == "ellipse") {
            shape = std::unique_ptr<Shape>(new Ellipse(geo, sz));
        }
        else if (type == "point") {
            shape = std::unique_ptr<Shape>(new Point(geo, sz));
        }
        else {
            CHECK(0) << "unknown shape: " << type;
        }
        auto plabel = spec.find("label");
        if (plabel != spec.end()) { // has label
            // might be a single number or an array
            if (plabel->is_array()) {
                cv::Scalar color(0,0,0,0);
                int i = 0;
                for (auto it = plabel->begin(); it != plabel->end(); ++it) {
                    color[i] = it->get<float>();
                    ++i;
                }
                shape->color = color;
            }
            else {
                float v = plabel->get<float>();
                shape->color = cv::Scalar(v, v, v, v);
            }
        }
        auto ptag = spec.find("tag");
        if (ptag != spec.end()) { // has label
            shape->tag = ptag->get<float>();
        }
        return shape;
    }
}
