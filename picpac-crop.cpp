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
        float scale;
        Config (): scale(1.0) {
        }
    };
private:
    Config config;
    ImageEncoder encoder;
    FileWriter db;
public:
    Splitter (Config const &c): config(c), encoder(".jpg"), db(config.path) {
    }
    ~Splitter () {
    }
    void add (Record const &rec) {
        Annotation anno(rec.field_string(1));
        if (anno.shapes.size() == 0) {
            return;
        }
        cv::Mat image = decode_buffer(rec.field(0), -1);
        string f0;
        string f1;
        for (unsigned i = 0; i < anno.shapes.size(); ++i) {
            auto shape = anno.shapes[i];
            cv::Rect_<float> bb;
            shape->bbox(&bb);
            cv::Rect bbi(bb.x * image.cols, bb.y * image.rows, bb.width * image.cols, bb.height * image.rows);
            cv::Mat roi = image(bbi);
            if (config.scale != 1) {
                cv::Mat tmp;
                cv::resize(roi, tmp, cv::Size(), config.scale, config.scale);
                roi = tmp;
            }
            encoder.encode(roi, &f0);
            Record rec(0, f0);
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
        ("scale", po::value(&config.scale), "")
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

