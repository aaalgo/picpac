#include "picpac-image.h"

namespace picpac {

    cv::Point round (cv::Point2f p) {
        return cv::Point(std::round(p.x), std::round(p.y));
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
            cv::circle(*m, round(controls[0]), opt.point_radius, color, opt.thickness, opt.line_type, opt.shift);
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
        }

        virtual std::unique_ptr<Shape> clone () const {
            return std::unique_ptr<Shape>(new Rectangle(*this));
        }

        virtual void render (cv::Mat *m, RenderOptions const &opt) const {
            cv::rectangle(*m, round(controls[0]), round(controls[1]), color, opt.thickness, opt.line_type, opt.shift);
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
                controls.emplace_back(p.at("x").get<float>(), p.at("y").get<float>());
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
                cv::fillPoly(*m, &pps, &nps, 1, color, opt.line_type, opt.shift);
            }
            else {
                cv::polylines(*m, &pps, &nps, 1, true, color, opt.thickness, opt.line_type, opt.shift);
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
        return shape;
    }
}
