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
    string code;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("input,i", po::value(&input_path), "")
    ("output,o", po::value(&output_path), "")
    ("compact", "")
    ("index-label2", "")
    ("transcode", po::value(&code), "")
    ;

    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || input_path.empty() || output_path.empty()) {
        cerr << "Usage:  picpac-dupe [options] <input> <output>" << endl;
        cerr << desc;
        cerr << endl;
        return 1;
    }
    int flags = 0;
    if (vm.count("compact")) flags |= picpac::FileWriter::COMPACT;
    if (vm.count("index-l2")) flags |= picpac::FileWriter::INDEX_LABEL2;
    google::InitGoogleLogging(argv[0]);
    picpac::FileReader in(input_path);
    picpac::FileWriter db(output_path, flags);
    vector<picpac::Locator> ll;
    in.ping(&ll);
    size_t count = 0;
    picpac::ImageEncoder encoder(code);
    for (auto const &l: ll) {
        Record rec;
        in.read(l, &rec);
        if (!code.empty()) {
            cv::Mat image = picpac::decode_buffer(rec.field(0), -1);
            string buf;
            encoder.encode(image, &buf);
            rec.replace(0, buf);
        }
        db.append(rec);
        ++count;
    }
    LOG(INFO) << "Loaded " << count << " samples.";

    return 0;
}


