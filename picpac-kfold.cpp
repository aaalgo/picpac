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

int main(int argc, char const* argv[]) {
    fs::path input_path;
    fs::path output_path;
    Stream::Config config;
    unsigned max_test;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("input", po::value(&input_path), "")
        ("output", po::value(&output_path), "")
        ("seed", po::value(&config.seed), "")
        ("split", po::value(&config.split)->default_value(5), "")
        ("fold", po::value(&config.split_fold)->default_value(0), "")
        ("stratify", po::value(&config.stratify), "")
        ("max-test", po::value(&max_test)->default_value(0), "0 for no limit")
        ;

    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help") || input_path.empty() || output_path.empty()) {
        cout << "Usage:" << endl;
        cout << "\tpicpac-split <output> <input> [<input> ...]" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }

    fs::path train_path(output_path);
    train_path += ".train";
    fs::path test_path(output_path);
    test_path += ".test";
    FileWriter train(train_path, FileWriter::COMPACT);
    FileWriter test(test_path, FileWriter::COMPACT);

    config.loop = false;
    config.shuffle = true;
    {
        config.split_negate = false;
        Stream  str(input_path, config);
        for (;;) {
            Record rec;
            try {
                str.read_next(&rec);
            }
            catch (EoS const &) {
                break;
            }
            train.append(rec);
        }
    }
    {
        config.split_negate = true;
        Stream  str(input_path, config);
        for (unsigned cc = 0;; ++cc) {
            Record rec;
            try {
                str.read_next(&rec);
            }
            catch (EoS const &) {
                break;
            }
            if ((max_test == 0) || (cc < max_test)) {
                test.append(rec);
            }
            else {
                train.append(rec);
            }
        }
    }

    return 0;
}

