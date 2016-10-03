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

class Context {

public:
    struct Config {
        string path;
        int size;
        Config (): size(240){
        }
    };
private:
    Config config;
    ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::min, ba::tag::max, ba::tag::variance > > acc;
    ImageEncoder encoder;
    FileWriter db;
public:
    Context (Config const &c): config(c), encoder(".jpg"), db(config.path)
    {
    }
    ~Context () {
    }

    void add (Record const &rec) {
        cv::Mat image = decode_buffer(rec.field(0), -1);
        if (image.cols <= config.size
                && image.rows <= config.size) {
            // downsize
            db.append(rec);
            return;
        }
        float r = std::min(1.0 * config.size / image.cols,
                           1.0 * config.size / image.rows);
        cv::resize(image, image, cv::Size(), r, r);
        string f0;
        encoder.encode(image, &f0);
        if (rec.meta().width < 2) {
            Record recx(0, f0);
            recx.meta().copy(rec.meta());
            db.append(recx);
        }
        else if (rec.meta().width == 2) {
            Record recx(0, f0, rec.field_string(1));
            recx.meta().copy(rec.meta());
            db.append(recx);
        }
    }
};

int main(int argc, char const* argv[]) {
    Context::Config config;
    fs::path input_path;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("input", po::value(&input_path), "")
        ("output", po::value(&config.path), "")
        ("size", po::value(&config.size), "")
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

    Context ctx(config);
    picpac::IndexedFileReader db(input_path);
    db.loop(std::bind(&Context::add, &ctx, placeholders::_1));
    return 0;
}

