#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include "picpac-cv.h"

using namespace std;
using namespace picpac;
namespace ba = boost::accumulators;

class Splitter {

public:
    struct Config {
        string path;
        string codec;
        string anno_codec;
#if 0
        string bg_path;
#endif
        int size;
        int width; 
        int height; 
        bool no_scale;
        // for grid splitting
        float grid_size;
        float grid_step;
        float grid_scale;
        bool image_annotation;
        Config (): size(50), width(100), height(100), no_scale(false), grid_size(240), grid_step(200), grid_scale(1), image_annotation(false) {
        }
    };
private:
    Config config;
    ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::min, ba::tag::max, ba::tag::variance > > acc;
    ImageEncoder encoder;
    ImageEncoder anno_encoder;
    FileWriter db;
#if 0
    FileWriter *bg;
#endif
    void add_roi (cv::Mat image,
                  cv::Rect roi,
                  Annotation const &anno) {
        cv::Rect_<float> zoom(1.0 * roi.x / image.cols,
                      1.0 * roi.y / image.rows,
                      1.0 * roi.width / image.cols,
                      1.0 * roi.height / image.rows);
        cv::Mat out = image(roi);
        if (out.total() == 0) {
            LOG(ERROR) << "skipping empty image WH="
                       << image.cols << ',' << image.rows
                       << " ROI XYWH=" << roi.x << ',' << roi.y << ',' << roi.width << ',' << roi.height;
            return;
        }
        Annotation anno_out;
        int label = 0;
        for (unsigned j = 0; j < anno.shapes.size(); ++j) {
            auto shapex = anno.shapes[j];
            cv::Rect_<float> bbx;
            shapex->bbox(&bbx);
            cv::Rect_<float> sect = bbx & zoom;
            if (sect.area() > 0) {
                anno_out.shapes.push_back(shapex->clone());
                //label = 1;
            }
        }
        string f0;
        string f1;
        anno_out.zoom(zoom);
        encoder.encode(out, &f0);
        anno_out.dump(&f1);
        Record rec(label, f0, f1);
        db.append(rec);
    }

    void add_roi_image_anno (cv::Mat image,
                  cv::Rect roi,
                  cv::Mat anno) {
        cv::Rect_<float> zoom(1.0 * roi.x / image.cols,
                      1.0 * roi.y / image.rows,
                      1.0 * roi.width / image.cols,
                      1.0 * roi.height / image.rows);
        cv::Mat out = image(roi);
        cv::Mat out_anno = anno(roi);
        string f0;
        string f1;
        encoder.encode(out, &f0);
        anno_encoder.encode(out_anno, &f1);
        int label = 0;
        Record rec(label, f0, f1);
        db.append(rec);
    }

    static void adjust_grid_size(int size, int *patch, int *step, int *nsteps) {
        if (*patch >= size) {
            *patch = size;
            *step = size;
            *nsteps = 1;
            return;
        }
        int s_size = size - *patch;
        int n = (s_size + *step - 1) / *step;
        int miss = (n * (*step) - s_size) / n;
        *nsteps = n;
        *step -= miss;
    }
public:
    Splitter (Config const &c): config(c), encoder(config.codec), anno_encoder(config.anno_codec), db(config.path)
#if 0
                                , bg(nullptr) 
#endif
    {
#if 0
        if (config.bg_path.size()) {
            bg = new FileWriter(config.bg_path);
        }
#endif
    }
    ~Splitter () {
        cout << "min: " << ba::min(acc) << endl;
        cout << "mean: " << ba::mean(acc) << endl;
        cout << "max: " << ba::max(acc) << endl;
#if 0
        delete bg;
#endif
    }

    void add (Record const &rec) {
        CHECK(!config.image_annotation);
        if ((rec.meta().width < 2) || (rec.meta().fields[1].size == 0)) {
#if 0
            if (bg) {
                bg->append(rec);
            }
#endif
            return;
        }
        Annotation anno(rec.field_string(1));
        if (anno.shapes.size() == 0) {
#if 0
            if (bg) {
                bg->append(rec);
            }
#endif
            return;
        }
        cv::Mat image = decode_buffer(rec.field(0), -1);
#if 0
        cv::Mat image_bg;
        if (bg) image_bg = image.clone();
#endif
        // TODO: add support for multiple bounding boxes
        //CHECK(anno.shapes.size() == 1);
        for (unsigned i = 0; i < anno.shapes.size(); ++i) {
            auto shape = anno.shapes[i];
            cv::Rect_<float> bb;
            shape->bbox(&bb);
#if 0
            if (bg) {
                shape->draw(&image_bg, cv::Scalar(0,0,0), CV_FILLED);
            }
#endif
            float scale = config.size / std::sqrt(bb.width * bb.height * image.rows * image.cols);
            acc(scale);
            cv::Mat scaled;
            cv::Rect roi;
            int dw = 0, dh = 0;
            if (config.no_scale) {
                scaled = image;
                roi = cv::Rect(bb.x * scaled.cols,
                             bb.y * scaled.rows,
                             bb.width * scaled.cols,
                             bb.height * scaled.rows);
                float factor = 1.0 * std::sqrt(1.0 * roi.width * roi.height) / config.size;
                int w = factor * config.width;
                int h = factor * config.height;
                if (w > roi.width) dw = w - roi.width;
                if (h > roi.height) dh = h - roi.height;
            }
            else {
                cv::resize(image, scaled, cv::Size(), scale, scale);
                //    |       |- width -|    |
                //    |----- cols -----------|
                roi = cv::Rect(bb.x * scaled.cols,
                             bb.y * scaled.rows,
                             bb.width * scaled.cols,
                             bb.height * scaled.rows);
                CHECK(roi.width > 0);
                CHECK(roi.height > 0);
                /*
                CHECK(roi.width <= config.width);
                CHECK(roi.height <= config.height);
                dw = config.width - roi.width;
                dh = config.height - roi.height;
                */
                dw = 0;
                if (roi.width <= config.width) {
                    dw = config.width - roi.width;
                }
                dh = 0;
                if (roi.height <= config.height) {
                    dh = config.height - roi.height;
                }
            }
            roi.x -= dw / 2;
            roi.width += dw;
            roi.y -= dh / 2;
            roi.height += dh;
            if (roi.width > scaled.cols) roi.width = scaled.cols;
            if (roi.height > scaled.rows) roi.height = scaled.rows;
            if (roi.x < 0) roi.x = 0;
            if (roi.y < 0) roi.y = 0;
            if (roi.x + roi.width > scaled.cols) roi.x = scaled.cols - roi.width;
            if (roi.y + roi.height > scaled.rows) roi.y = scaled.rows - roi.height;
            CHECK(roi.x >= 0);
            CHECK(roi.y >= 0);

            CHECK(scaled.cols > 0);
            CHECK(scaled.rows > 0);
            add_roi(scaled, roi, anno);
        }
#if 0
        if (bg) {
            encoder.encode(image_bg, &f0);
            Record rec(0, f0);
            bg->append(rec);
        }
#endif
    }

    void add_grid (Record const &rec) {
        Annotation anno;
        if ((rec.meta().width >= 2) && (rec.meta().fields[1].size > 0) && !config.image_annotation) {
            Annotation annox(rec.field_string(1));
            anno.shapes.swap(annox.shapes);
        }
        cv::Mat image = decode_buffer(rec.field(0), -1);
        cv::Mat anno_image;
        if (config.image_annotation) {
            anno_image = decode_buffer(rec.field(1), -1);
        }
        if (config.grid_scale != 1) {
            cv::Mat scaled;
            cv::resize(image, scaled, cv::Size(), config.grid_scale, config.grid_scale);
            image = scaled;
            if (config.image_annotation) {
                scaled = cv::Mat();
                cv::resize(anno_image, scaled, cv::Size(), config.grid_scale, config.grid_scale);
                anno_image = scaled;
            }
        }
        int sx = config.grid_size;  // patch size
        int dx = config.grid_step;  // step
        int nx;                     // # steps
        int sy = config.grid_size;
        int dy = config.grid_step;
        int ny;
        adjust_grid_size(image.rows, &sy, &dy, &ny);
        adjust_grid_size(image.cols, &sx, &dx, &nx);
        cv::Rect roi;
        roi.width = sx;
        roi.height = sy;
        for (int y = 0; y < ny; ++y) {
            roi.y = y * dy;
            if (roi.y + sy > image.rows) {
                roi.y = image.rows - sy;
            }
            for (int x = 0; x < nx; ++x) {
                roi.x = x * dx;
                if (roi.x + sx > image.cols) {
                    roi.x = image.cols - sx;
                }
                if (config.image_annotation) {
                    add_roi_image_anno(image, roi, anno_image);
                }
                else {
                    add_roi(image, roi, anno);
                }
            }
        }
    }
};

int main(int argc, char const* argv[]) {
    Splitter::Config config;
    fs::path input_path;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("input", po::value(&input_path), "")
        ("output", po::value(&config.path), "")
#if 0
        ("bg", po::value(&config.bg_path), "")
#endif
        ("no-scale", po::value(&config.no_scale), "")
        ("width", po::value(&config.width), "")
        ("height", po::value(&config.height), "")
        ("size", po::value(&config.size), "")
        ("grid-size", po::value(&config.grid_size), "")
        ("grid-step", po::value(&config.grid_step), "")
        ("grid-scale", po::value(&config.grid_scale), "")
        ("grid", "")
        ("image-annotation", "")
        ("codec", po::value(&config.codec)->default_value(".jpg"), "use .tiff for 16-bit images") 
        ("anno-codec", po::value(&config.anno_codec)->default_value(".png"), "not used json annotation") 
        ;

    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);
    p.add("bg", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help") || input_path.empty() || config.path.empty()) {
        cout << "Usage:" << endl;
        cout << "\tpicpac-stat ... <db>" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }
    if (vm.count("image-annotation")) config.image_annotation = true;

    Splitter splitter(config);
    picpac::IndexedFileReader db(input_path);
    if (vm.count("grid")) {
        db.loop(std::bind(&Splitter::add_grid, &splitter, placeholders::_1));
    }
    else {
        db.loop(std::bind(&Splitter::add, &splitter, placeholders::_1));
    }
    return 0;
}

