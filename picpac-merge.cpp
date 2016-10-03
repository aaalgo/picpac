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
    FileWriter db;
public:
    Context (fs::path const &path): db(path)
    {
    }
    ~Context () {
    }

    void add (Record const &rec) {
        db.append(rec);
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
        cout << "\tpicpac-merge <output> <input> [<input> ...]" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }

    Context ctx(output_path);
    for (auto const &input_path: input_paths) {
        picpac::IndexedFileReader db(input_path);
        db.loop(std::bind(&Context::add, &ctx, placeholders::_1));
    }
    return 0;
}

