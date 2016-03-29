#include <boost/program_options.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include "picpac.h"
#include "picpac-cv.h"
#include "picpac-util.h"

using namespace std;
using namespace boost;
using namespace picpac;

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    fs::path input_path;
    fs::path output_path;
    int max_size;
    int resize;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("input,i", po::value(&input_path), "")
    ("output,o", po::value(&output_path), "")
    ("max", po::value(&max_size)->default_value(800), "")
    ("resize", po::value(&resize)->default_value(-1), "")
    /*
    ("gray", "")
    ("max", po::value(&max_size)->default_value(max_size), "")
    ("log-level,v", po::value(&FLAGS_minloglevel)->default_value(1), "")
    */
    ;

    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || input_path.empty() || output_path.empty()) {
        cerr << desc;
        return 1;
    }
    //if (vm.count("gray")) gray = true;

    google::InitGoogleLogging(argv[0]);
    picpac::FileWriter db(output_path);
    CachedDownloader downloader;
    ImageReader imreader(max_size, resize);
    fs::ifstream is(input_path.c_str());
    string line;
    int count = 0;
    while (getline(is, line)) {
        vector<string> ss;
        split(ss, line, is_any_of("\t"), token_compress_off);
        if (ss.size() != 2) {
            cerr << "Bad line: " << line << endl;
            continue;
        }
        try {
            fs::path path = downloader.download(ss[0]);
            string data;
            imreader.read(path, &data);
            Record record(0, data, ss[1]);
            db.append(record);
            ++count;
        }
        catch (...) {
            LOG(ERROR) << "Fail to load " << ss[0];
        }
    }
    LOG(INFO) << "Loaded " << count << " samples.";

    return 0;
}


