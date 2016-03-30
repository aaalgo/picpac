#include <chrono>
#include <mutex>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <served/served.hpp>
#include <served/plugins.hpp>
#include <magic.h>
#include "picpac-cv.h"
#include "rfc3986.h"

/*
static inline void EncodeImage (const cv::Mat &image, std::string *binary) {
    std::vector<uint8_t> buffer;
    cv::imencode(".jpg", image, buffer);
    if (buffer.empty()) {
        binary->clear();
        return;
    }
    binary->resize(buffer.size());
    char *from = reinterpret_cast<char *>(&buffer[0]);
    std::copy(from, from + buffer.size(), &(*binary)[0]);
}
*/

using namespace std;
using namespace picpac;

class no_throw {
    typedef function<void(served::response &res, const served::request &req)> callback_type;
    callback_type cb;
public:
    no_throw (callback_type cb_): cb(cb_) {
    }
    void operator () (served::response &res, const served::request &req) {
        bool ok = false;
        try {
            cb(res, req);
            ok = true;
        }
        catch (runtime_error const &e) {
            res.set_header("PicPacError", e.what());
        }
        catch (...) {
            res.set_header("PicPacError", "unknown error");
        }
        res.set_status(ok ? 200 : 500);
    }
};

int main(int argc, char const* argv[]) {
    string address;
    string port;
    int threads;
    fs::path db_path;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("address", po::value(&address)->default_value("0.0.0.0"), "")
        ("port", po::value(&port)->default_value("18888"), "")
        ("threads", po::value(&threads)->default_value(4), "")
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
        cout << "\tserver ... <db>" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }

    // Create a multiplexer for handling requests
    served::multiplexer mux;

    picpac::IndexedFileReader db(db_path);

    magic_t cookie = magic_open(MAGIC_MIME_TYPE);
    CHECK(cookie);
    magic_load(cookie, NULL);

    default_random_engine rng;
    // GET /hello
    mux.handle("/hello")
        .get([](served::response &res, const served::request &req) {
            res << "Hello world!";
        });
    mux.handle("/file")
        .get(no_throw([&db, &cookie](served::response &res, const served::request &req) {
            rfc3986::Form query(req.url().query());
            int id = lexical_cast<int>(query["id"]);
            Record rec;
            db.read(id, &rec);
            string buf = rec.field_string(0);
            char const *mime = magic_buffer(cookie, &buf[0], buf.size());
            res.set_header("Content-Type", mime);
            res.set_body(buf);
        }));
    mux.handle("/anno")
        .get(no_throw([&db, &rng](served::response &res, const served::request &req) {
            rfc3986::Form query(req.url().query());
            ImageLoader::Config conf;
            conf.annotate = ImageLoader::ANNOTATE_JSON;
            conf.anno_type = CV_8UC3;
            conf.anno_copy = true;
            conf.anno_color = cv::Scalar(0, 0, 255);
            conf.anno_thickness = query.get<int>("th", 2);
            conf.pert_color = cv::Scalar(20,20,20);
            conf.pert_angle = 20;
            conf.pert_min_scale = 0.8;
            conf.pert_max_scale = 1.2;
            conf.pert_hflip = conf.pert_vflip = true;
            conf.perturb = query.get<int>("pt", 0);
            ImageLoader loader(conf);
            ImageLoader::PerturbVector pv;
            int id = lexical_cast<int>(query["id"]);
            ImageLoader::Value v;
            loader.sample(rng, &pv);
            loader.load([&db, id](Record *r){db.read(id, r);}, pv, &v);
            ImageEncoder encoder;
            string buf;
            encoder.encode(v.annotation, &buf);
            res.set_header("Content-Type", "image/jpeg");
            res.set_body(buf);
        }));
    mux.use_after(served::plugin::access_log);
    LOG(INFO) << "listening at " << address << ':' << port;
    served::net::server server(address, port, mux);
    LOG(INFO) << "running server with " << threads << " threads.";
    server.run(threads);
    magic_close(cookie);
    return 0;
}

