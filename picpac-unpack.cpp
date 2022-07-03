#include <fstream>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include "picpac-cv.h"

using namespace std;
using namespace picpac;

int main(int argc, char const* argv[]) {
    ImageStream::Config config;
    config.loop = false;
    config.channels = 1;
    config.stratify = false;
    config.annotate = "none";

    fs::path db_path;
    fs::path dir_path;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("db", po::value(&db_path), "")
        ("dir", po::value(&dir_path), "")
        ;

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
    int c = 0;
    if (!dir_path.empty()) {
        fs::create_directories(dir_path);
    }
    picpac::IndexedFileReader db(db_path);
    db.loop([&c, dir_path](Record const &rec){
        fs::path o = dir_path / (lexical_cast<string>(c) + ".jpg");
        ofstream os(o, ios::binary);
        auto f = rec.field(0);
        os.write(boost::asio::buffer_cast<char const *>(f),
                 boost::asio::buffer_size(f));
        ++c;
    });
    cout << c << " samples loaded." << endl;

    return 0;
}

