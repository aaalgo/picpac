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
    int threads = 1;
    fs::path db_path;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("address", po::value(&address)->default_value("0.0.0.0"), "")
        ("port", po::value(&port)->default_value("18888"), "")
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
    default_random_engine rng;

    magic_t cookie = magic_open(MAGIC_MIME_TYPE);
    CHECK(cookie);
    magic_load(cookie, NULL);

    // GET /hello
    mux.handle("/l")
        .get(no_throw([&db, &cookie](served::response &res, const served::request &req) {
            rfc3986::Form query(req.url().query());
            rfc3986::Form trans;
            // transfer all applicable image parameters to trans
            // so we can later use that for image display
#define PICPAC_CONFIG_UPDATE(C,P) \
            { auto it = query.find(#P); if (it != query.end()) trans.insert(*it);}
            PICPAC_CONFIG_UPDATE_ALL(0);
#undef PICPAC_CONFIG_UPDATE
            string ext = trans.encode(true);
            int count = query.get<int>("count", 20);
            string anno = query.get<string>("anno", "");
            res << "<html><body><table><tr><th>Image</th></tr>";
            for (unsigned i = 0; i < count; ++i) {
                int id = rand() % db.size();
                res << "<tr><td><img src=\"/image?id="
                    << lexical_cast<string>(id) << ext
                    << "\"></img></td></tr>";
            }
            res << "</table></body></html>";
        }));
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
    mux.handle("/image")
        .get(no_throw([&db, &rng](served::response &res, const served::request &req) {
            rfc3986::Form query(req.url().query());
            PICPAC_CONFIG conf;
            conf.anno_color3 = 255;
            conf.anno_copy = true;
            conf.anno_thickness = 2;
            conf.pert_color1 = 20;
            conf.pert_color2 = 20;
            conf.pert_color3 = 20;
            conf.pert_angle = 20;
            conf.pert_min_scale = 0.8;
            conf.pert_max_scale = 1.2;
#define PICPAC_CONFIG_UPDATE(C,P) C.P = query.get<decltype(C.P)>(#P, C.P)
            PICPAC_CONFIG_UPDATE_ALL(conf);
#undef PICPAC_CONFIG_UPDATE
            ImageLoader loader(conf);
            ImageLoader::PerturbVector pv;
            int id = query.get<int>("id", rng() % db.size());
            ImageLoader::Value v;
            loader.sample(rng, &pv);
            loader.load([&db, id](Record *r){db.read(id, r);}, pv, &v);
            ImageEncoder encoder;
            string buf;
            cv::Mat image = v.image;
            if (conf.annotate.size()) {
                image = v.annotation;
            }
            encoder.encode(image, &buf);
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

