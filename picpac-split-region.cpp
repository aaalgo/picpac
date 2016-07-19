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
        int size;
        int width; 
        int height; 
        bool no_scale;
        Config (): size(50), width(240), height(240), no_scale(false) {
        }
    };
private:
    Config config;
    ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::min, ba::tag::max, ba::tag::variance > > acc;
    ImageEncoder encoder;
    FileWriter db;
public:
    Splitter (Config const &c): config(c), encoder(".jpg"), db(config.path) {
    }
    void add (Record const &rec) {
        if (rec.meta().width < 2) {
            return;
        }
        cv::Mat image = decode_buffer(rec.field(0), -1);
        Annotation anno(rec.field_string(1));
        // TODO: add support for multiple bounding boxes
        CHECK(anno.shapes.size() == 1);
        string f0;
        string f1;
        for (unsigned i = 0; i < anno.shapes.size(); ++i) {
            auto shape = anno.shapes[i];
            cv::Rect_<float> bb;
            shape->bbox(&bb);
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
                CHECK(roi.width <= config.width);
                CHECK(roi.height <= config.height);
                dw = config.width - roi.width;
                dh = config.height - roi.height;
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
            cv::Rect_<float> zoom(1.0 * roi.x / scaled.cols,
                          1.0 * roi.y / scaled.rows,
                          1.0 * roi.width / scaled.cols,
                          1.0 * roi.height / scaled.rows);
            cv::Mat out = scaled(roi);
            Annotation anno_out;
            cout << zoom.x << ' ' << zoom.y << ' ' << zoom.width << ' ' << zoom.height << ' ' << scaled.cols << ' ' << scaled.rows << endl;
            anno_out.shapes.push_back(shape->clone());
            anno_out.dump(&f1);
            cout << f1 << endl;
            anno_out.zoom(zoom);
            encoder.encode(out, &f0);
            anno_out.dump(&f1);
            cout << f1 << endl;
            Record rec(1, f0, f1);
            db.append(rec);
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
        ("no-scale", po::value(&config.no_scale), "")
        ;

    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

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

    Splitter splitter(config);
    picpac::IndexedFileReader db(input_path);
    db.loop(std::bind(&Splitter::add, &splitter, placeholders::_1));
    return 0;
}

