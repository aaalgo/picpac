#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include "picpac-cv.h"

using namespace std;
using namespace picpac;
namespace ba = boost::accumulators;

int main(int argc, char const* argv[]) {
    BatchImageStream::Config config;
    fs::path db_path;
    config.loop = false;
    config.split = 0;
    config.shuffle = false;
    config.annotate = "json";
    BatchImageStream::Config config2 = config;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        (",m", po::value(&config2.anno_min_ratio)->default_value(0.05), "")
        ("db", po::value(&db_path), "")
        ;
#define PICPAC_CONFIG_UPDATE(C,p) desc.add_options()(#p, po::value(&C.p)->default_value(C.p), "")
    PICPAC_CONFIG_UPDATE_ALL(config);
#undef PICPAC_CONFIG_UPDATE

    po::positional_options_description p;
    p.add("db", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help") || db_path.empty()) {
        cout << "Usage:" << endl;
        cout << "\tpicpac-stat ... <db>" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }
    ImageStream db(db_path, config);
    ImageStream db2(db_path, config2);
    ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::min, ba::tag::max, ba::tag::variance > > acc, acc2;
    for (;;) {
        try {
            ImageStream::Value v(db.next());
            ImageStream::Value v2(db2.next());
            if (v.annotation.data) {
                float roi = cv::sum(v.annotation)[0];
                float roi2 = cv::sum(v2.annotation)[0];
                //cout << roi << ' ' << v.annotation.total() << endl;
                acc(roi/v.annotation.total());
                if (roi > 0) {
                    acc2(std::sqrt(roi2/roi));
                }
            }
        }
        catch (EoS const &) {
            break;
        }
    }
    cout << "ratio: " << ba::mean(acc) << " +/- " << std::sqrt(ba::variance(acc)) << endl;
    cout << "scale: " << ba::mean(acc2) << " +/- " << std::sqrt(ba::variance(acc2)) << endl;
    /*
    cout << ba::min(acc) << endl;
    cout << ba::mean(acc) << endl;
    cout << ba::max(acc) << endl;
    cout << std::sqrt(ba::variance(acc)) << endl;
    */


    return 0;
}

