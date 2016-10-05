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
    bool ra;
    int label;
public:
    Context (fs::path const &path, bool ra_): db(path), ra(ra_), label(-1)
    {
    }
    ~Context () {
    }

    void set_label (int l) {
        label = l;
    }

    void add (Record &rec) {
        if (label >= 0) {
            rec.meta().label = label;
        }
        if (ra) {
            rec.meta().width = 1;
        }
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
        ("ra", "")
        ("la", "")
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

    bool ra = vm.count("ra");
    bool la = vm.count("la");

    int l = 0;
    Context ctx(output_path, ra);
    for (auto const &input_path: input_paths) {
        if (la) ctx.set_label(l);
        picpac::IndexedFileReader db(input_path);
        db.loop(std::bind(&Context::add, &ctx, placeholders::_1));
        ++l;
    }
    return 0;
}

