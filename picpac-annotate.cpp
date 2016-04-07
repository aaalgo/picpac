#include <unordered_map>
#include <boost/program_options.hpp>
#include "picpac-cv.h"

using namespace std;
using namespace picpac;

int main(int argc, char const* argv[]) {
    ImageStream::Config config;
    config.loop = false;
    config.stratify = false;
    config.anno_color1 = 255;
    config.anno_color3 = 255;
    config.anno_copy = true;
    config.anno_thickness = 1;

    int max_size;
    fs::path db_path;
    fs::path dir_path;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("max", po::value(&max_size)->default_value(-1), "")
        ("annotate,a", po::value(&config.annotate)->default_value("json"), "")
        ("db", po::value(&db_path), "")
        ("dir", po::value(&dir_path), "")
        ;

    po::positional_options_description p;
    p.add("db", 1);
    p.add("dir", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help") || db_path.empty() || dir_path.empty()) {
        cout << "Usage:" << endl;
        cout << "\tpicpac-stat ... <db>" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }
    ImageStream db(db_path, config);
    fs::create_directories(dir_path);
    int c = 0;
    for (;;) {
        try {
            ImageStream::Value v(db.next());
            cv::Mat image = v.annotation;
            if (max_size > 0) {
                cv::Mat tmp;
                LimitSize(image, max_size, &tmp);
                image = tmp;
            }
            fs::path o = dir_path / (lexical_cast<string>(c++) + ".jpg");
            cv::imwrite(o.native(), image);
        }
        catch (EoS const &) {
            break;
        }
    }

    return 0;
}

