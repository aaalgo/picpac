#include <boost/program_options.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/progress.hpp>
#include "picpac.h"
#include "picpac-cv.h"
#include "picpac-util.h"

using namespace std;
using namespace boost;
using namespace picpac;

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    fs::path db_path;
    unsigned N;
    BatchImageStream::Config config;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("db", po::value(&db_path), "")
    (",N", po::value(&N)->default_value(16000), "")
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
        cerr << desc;
        return 1;
    }

    google::InitGoogleLogging(argv[0]);
    cv::setNumThreads(1);

    ImageStream stream(db_path, config);

    progress_display progress(N, cerr);
    for (unsigned i = 0; i < N; ++i) {
        auto v(stream.next());
        CHECK(v.image.data);
        if (config.annotate.size()) {
            CHECK(v.annotation.data);
        }
        ++progress;
    }


    return 0;
}


