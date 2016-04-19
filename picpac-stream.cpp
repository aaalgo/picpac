#include <unordered_map>
#include <boost/program_options.hpp>
#include "picpac-cv.h"

using namespace std;
using namespace picpac;

int main(int argc, char const* argv[]) {
    BatchImageStream::Config config;
    unsigned max;
    float scale;
    fs::path db_path;
    fs::path dir_path;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("max", po::value(&max)->default_value(100), "")
        ("scale", po::value(&scale)->default_value(1), "")
        ("db", po::value(&db_path), "")
        ("dir", po::value(&dir_path), "")
        ;
#define PICPAC_CONFIG_UPDATE(C,p) desc.add_options()(#p, po::value(&C.p)->default_value(C.p), "")
    PICPAC_CONFIG_UPDATE_ALL(config);
#undef PICPAC_CONFIG_UPDATE

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
    for (unsigned i = 0; i < max; ++i) {
        try {
            ImageStream::Value v(db.next());
            fs::path ip = dir_path / (lexical_cast<string>(c++) + ".jpg");
            if (scale != 1) {
                cv::resize(v.image, v.image, cv::Size(), scale, scale);
                if (v.annotation.data) {
                    cv::resize(v.annotation, v.annotation, cv::Size(), scale, scale);
                }
            }
            cv::imwrite(ip.native(), v.image);
            if (v.annotation.data) {
                fs::path ap = dir_path / (lexical_cast<string>(c++) + "a.jpg");
                cv::imwrite(ap.native(), v.annotation);
            }
        }
        catch (EoS const &) {
            break;
        }
    }

    return 0;
}

