#include <unordered_map>
#include <boost/program_options.hpp>
#include "picpac.h"

using namespace std;
using namespace picpac;

int main(int argc, char const* argv[]) {
    fs::path db_path;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("db", po::value(&db_path), "")
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

    IndexedFileReader db(db_path);
    std::unordered_map<int, int> cnt;
    cout << "TOTAL: " << db.size() << endl;
    bool cat = true;
    for (unsigned i = 0; i < db.size(); ++i) {
        float l = db.group(i);
        if (int(l) != l) {
            cat = false;
            break;
        }
        cnt[l] += 1;
    }
    if (cat) {
        vector<std::pair<int, int>> all(cnt.begin(), cnt.end());
        sort(all.begin(), all.end());
        for (auto const &p: all) {
            cout << p.first << ": " << p.second << endl;
        }
    }
    

    return 0;
}

