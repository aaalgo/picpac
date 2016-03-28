#include <atomic>
#include <iostream>
#include <fstream>
#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <json11.hpp>
#include "picpac.h"

using namespace std;
using namespace json11;
namespace fs = boost::filesystem;

struct Line {
    string path;
    float label;
};

int main (int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    namespace po = boost::program_options; 
    string root;
    string list_path;
    string out_path;
    unsigned threads;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("list", po::value(&list_path), "")
    ("root", po::value(&root), "")
    ("out", po::value(&out_path), "")
    ("threads,t", po::value(&threads)->default_value(0), "")
    ;   
    
    po::positional_options_description p;
    p.add("root", 1); 
    p.add("list", 1); 
    p.add("out", 1); 

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || !root.size() || !out_path.size()) {
        cerr << desc;
        return 1;
    }
    vector<Line> lines;
    {
        ifstream is(list_path);
        Line line;
        while (is >> line.path >> line.label) {
            lines.push_back(line);
        }
        LOG(INFO) << "Loaded " << lines.size() << " lines." << endl;
    }

    {
        boost::timer::auto_cpu_timer t;
        picpac::FileWriter dataset(out_path);
        std::atomic<unsigned> done(0);
        fs::path root_path(root);
#pragma omp parallel for
        for (unsigned i = 0; i < lines.size(); ++i) {
            picpac::Record rec;
            fs::path path = lines[i].path;
            /*
            rec.meta.label = lines[i].label;
            rec.meta.serial = lines[i].serial;
            */
            if (!root_path.empty()) {
                path = root_path / path;
            }
            cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
            if (image.total() == 0) {
                LOG(ERROR) << "fail to load image " << path;
                continue;
            }
            Json json = Json::object {
                {"path", lines[i].path},
            };
            string extra = json.dump();
            rec.pack(lines[i].label, path, extra);
#pragma omp critical
            dataset.append(rec);
            unsigned n = done.fetch_add(1);
            LOG_IF(INFO, n && ((n % 1000) == 0)) << n << '/' << lines.size() << " images.";
        }
    }
}

