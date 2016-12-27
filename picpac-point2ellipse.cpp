#include <json11.hpp>
#include <boost/program_options.hpp>
#include "picpac-cv.h"

namespace picpac {
    cv::Mat decode_buffer (const_buffer imbuf, int mode);
}


using namespace std;
using namespace json11;
using namespace picpac;

class Context {
    FileWriter db;
    int radius;
public:
    Context (fs::path const &path, int r): db(path), radius(r) {
    }
    ~Context () {
    }

    void add (Record &rec) {
        do {
            if (rec.size() <= 1) break;
            CHECK(rec.size() == 2);
            cv::Mat image = decode_buffer(rec.field(0), -1);
            string err;
            Json json = Json::parse(rec.field_string(1), err);
            int w = image.cols;
            int h = image.rows;
            bool update = false;
            Json::array shapes;
            for (auto const &shape: json["shapes"].array_items()) {
                if (shape["type"] == "point") {
                    update = true;
                    float x = shape["geometry"]["x"].number_value();
                    float y = shape["geometry"]["y"].number_value();
                    x *= w;
                    y *= h;
                    Json geo = Json::object{
                                {"x", 1.0 * (x - 1.0 * radius) / w},
                                {"y", 1.0 * (y - 1.0 * radius) / h},
                                {"width", 2.0 * radius / w},
                                {"height", 2.0 * radius / h}};
                    /*
                    Json ss = Json::object{
                            {"type", "ellipse"},
                            {"geometry", geo}};
                    shapes.push_back(ss);
                    */
                    shapes.emplace_back(Json::object{
                            {"type", "ellipse"},
                            {"geometry", geo}});
                }
                else {
                    shapes.push_back(shape);
                }
            }
            if (!update) break;
            cout << "<< " << rec.field_string(1) << endl;
            Json::object obj = json.object_items();
            obj["shapes"] = Json(shapes);
            json = Json(obj);
            string f2 = json.dump();
            cout << ">> " << f2 << endl;
            Record r(0, rec.field_string(0), f2);
            r.meta().copy(rec.meta());
            db.append(r);
            return;
        } while (false);
        db.append(rec);
    }
};

int main(int argc, char const* argv[]) {
    fs::path output_path;
    vector<fs::path> input_paths;
    int radius;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("input", po::value(&input_paths), "")
        ("output", po::value(&output_path), "")
        ("radius,r", po::value(&radius)->default_value(3), "")
        ;

    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help") || input_paths.empty() || output_path.empty()) {
        cout << "Usage:" << endl;
        cout << "\tpicpac-merge <output> <input> [<input> ...]" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }

    Context ctx(output_path, radius);
    for (auto const &input_path: input_paths) {
        picpac::IndexedFileReader db(input_path);
        db.loop(std::bind(&Context::add, &ctx, placeholders::_1));
    }
    return 0;
}

