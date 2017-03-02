#include <iostream>
#include <set>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include "picpac.h"

using namespace std;
using namespace picpac;

class IdFilter {
    set<int> ids;
    bool keep;
public:
    IdFilter (fs::path const &file, bool kp): keep(kp) {
        fs::ifstream is(file);
        int x;
        while (is >> x) {
            ids.insert(x);
        }
    }

    bool test (int id) {
        return keep == (ids.count(id) > 0);
    }
};

class Context: IdFilter {
    FileWriter db;
    int id;
public:
    Context (fs::path const &path, fs::path const &ids, bool keep): IdFilter(ids, keep), db(path), id(0)
    {
    }
    ~Context () {
    }

    void add (Record &rec) {
        if (test(id)) {
            db.append(rec);
        }
        ++id;
    }
};

int main(int argc, char const* argv[]) {
    fs::path ids_path;
    fs::path output_path;
    fs::path input_path;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("keep,k", po::value(&ids_path), "")
        ("exclude,x", po::value(&ids_path), "")
        ("input", po::value(&input_path), "")
        ("output", po::value(&output_path), "")
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
        cout << "\tpicpac-filter <-k ids| -x ids> <input> <output>" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }
    if ((vm.count("keep") + vm.count("exclude")) != 1) {
        cout << "you can use either --keep or --exclude but not neithor nor both" << endl;
        return 0;
    }

    bool keep = vm.count("keep") > 0;
    Context ctx(output_path, ids_path, keep);
    picpac::IndexedFileReader db(input_path);
    db.loop(std::bind(&Context::add, &ctx, placeholders::_1));
    return 0;
}

