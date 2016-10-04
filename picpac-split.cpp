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
    FileWriter db0;
    FileWriter db1;
public:
    Context (fs::path const &path0,
             fs::path const &path1): db0(path0), db1(path1)
    {
    }
    ~Context () {
    }

    void add (Record const &rec) {
        bool pos = true;
        if ((rec.meta().width < 2) || (rec.meta().fields[1].size == 0)) {
            pos = false;
        }
        Annotation anno(rec.field_string(1));
        if (anno.shapes.size() == 0) {
            pos = false;
        }
        if (pos) {
            db1.append(rec);
        }
        else {
            db0.append(rec);
        }
    }
};

int main(int argc, char const* argv[]) {
    fs::path output_path;
    vector<fs::path> input_paths;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("input", po::value(&input_paths), "")
        ("output", po::value(&output_path), "")
        ;

    po::positional_options_description p;
    p.add("output", 1);
    p.add("input", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help") || input_paths.empty() || output_path.empty()) {
        cout << "Usage:" << endl;
        cout << "\tpicpac-split <output> <input> [<input> ...]" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }

    fs::path path0 = output_path;
    path0 += ".0";
    fs::path path1 = output_path;
    path1 += ".1";
    Context ctx(path0, path1);
    for (auto const &input_path: input_paths) {
        picpac::IndexedFileReader db(input_path);
        db.loop(std::bind(&Context::add, &ctx, placeholders::_1));
    }
    return 0;
}

