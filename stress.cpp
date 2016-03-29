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
    ImageStream::Config config;
    unsigned N;
    int resize;
    string anno;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("db", po::value(&db_path), "")
    (",N", po::value(&N)->default_value(16000), "")
    ("anno", po::value(&anno)->default_value("none"), "")
    ("resize", po::value(&resize)->default_value(-1), "")
    ("threads", po::value(&config.threads), "")

    ;

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
    //if (vm.count("gray")) gray = true;
    if (anno == "none") {
        config.annotate = ImageLoader::ANNOTATE_NONE;
    }
    else if (anno == "json") {
        config.annotate = ImageLoader::ANNOTATE_JSON;
    }
    else if (anno == "image") {
        config.annotate = ImageLoader::ANNOTATE_IMAGE;
    }
    else CHECK(0) << "unknown annotate: " << anno;
    
    if (resize > 0) {
        config.resize = cv::Size(resize, resize);
    }

    google::InitGoogleLogging(argv[0]);

    ImageStream stream(db_path, config);

    progress_display progress(N, cerr);
    for (unsigned i = 0; i < N; ++i) {
        stream.next();
        ++progress;
    }


    return 0;
}


