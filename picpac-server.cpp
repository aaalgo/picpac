#define BOOST_SPIRIT_NO_PREDEFINED_TERMINALS
#include <chrono>
#include <mutex>
#include <vector>
#include <map>
#include <unordered_map>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <magic.h>
#include <json11.hpp>
#include <server_http.hpp>
#include "picpac-cv.h"
#include "rfc3986.h"
#include "bfdfs/bfdfs.h"

using namespace std;
using namespace picpac;

char const *version = PP_VERSION "-" PP_BUILD_NUMBER "," PP_BUILD_ID "," PP_BUILD_TIME;

extern char _binary_html_static_start;

static std::unordered_map<string, string> const EXT_MIME{
    {".html", "text/html"},
    {".css", "text/css"},
    {".js", "application/javascript"},
};
static std::string const DEFAULT_MIME("application/octet-stream");

void banner () {
    cout << "PicPac Server" << endl;
    cout << "Version: " << version << endl;
    cout << "https://github.com/aaalgo/picpac" << endl;
}

using namespace json11;

class HttpServer: public SimpleWeb::Server<SimpleWeb::HTTP> {
    picpac::IndexedFileReader db;
    bfdfs::Loader statics;
    default_random_engine rng;  // needs protection!!!TODO
    magic_t cookie;

#define HTTP_Q "(\\?.+)?$"

    string http_query (string const &path) const {
        auto p = path.find('?');
        if (p == path.npos) return "";
        return path.substr(p+1);
    }


    static string const &path2mime (string const &path) {
        do {
            auto p = path.rfind('.');
            if (p == path.npos) break;
            string ext = path.substr(p);
            auto it = EXT_MIME.find(ext);
            if (it == EXT_MIME.end()) break;
            return it->second;
        } while (false);
        return DEFAULT_MIME;
    }

    class no_throw {
        typedef function<void(shared_ptr<Response>, shared_ptr<Request>)> callback_type;
        callback_type cb;
    public:
        no_throw (callback_type cb_): cb(cb_) {
        }
        void operator () (shared_ptr<Response> res, shared_ptr<Request> req) {
#if 1
            bool ok = true;
            cb(res, req); 
            cout << "OK " << req->path << endl;
#else
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
#endif
            //res.set_status(ok ? 200 : 500);
        }
    };

public:
    HttpServer (fs::path const &db_path, unsigned short port, size_t num_threads=1)
        : Server(port, num_threads),
        db(db_path),
        statics(&_binary_html_static_start) {

        cookie = magic_open(MAGIC_MIME_TYPE);
        CHECK(cookie);
        magic_load(cookie, "/usr/share/misc/magic:/usr/local/share/misc/magic");

        resource["^/api/sample" HTTP_Q]["GET"] = no_throw([this](shared_ptr<HttpServer::Response> res, shared_ptr<HttpServer::Request> req) {
                rfc3986::Form query(http_query(req->path));
                /*
                rfc3986::Form trans;
                // transfer all applicable image parameters to trans
                // so we can later use that for image display
#define PICPAC_CONFIG_UPDATE(C,P) \
                { auto it = query.find(#P); if (it != query.end()) trans.insert(*it);}
                PICPAC_CONFIG_UPDATE_ALL(0);
#undef PICPAC_CONFIG_UPDATE
                string ext = trans.encode(true);
                */
                int count = query.get<int>("count", 20);
                //string anno = query.get<string>("anno", "");
                vector<int> ids(count);
                for (auto &id: ids) {
                    id = rand() % db.size();
                }
                Json json = Json::object {
                    {"samples", Json(ids)}
                };
                string json_str = json.dump();
                *res << "HTTP/1.1 200 OK\r\n";
                *res << "Content-Type: application/json\r\n";
                *res << "Content-Length: " << json_str.size() << "\r\n\r\n";
                *res << json_str;
            });
        resource["^/api/file" HTTP_Q]["GET"] =
            no_throw([this](shared_ptr<HttpServer::Response> res, shared_ptr<HttpServer::Request> req) {
                rfc3986::Form query(http_query(req->path));
                int id = lexical_cast<int>(query["id"]);
                Record rec;
                db.read(id, &rec);
                string buf = rec.field_string(0);
                char const *mime = magic_buffer(cookie, &buf[0], buf.size());
                *res << "HTTP/1.1 200 OK\r\n";
                *res << "Content-Type: " << mime << "\r\n";
                *res << "Content-Length: " << buf.size() << "\r\n\r\n";
                *res << buf;
            });
        resource["^/api/image" HTTP_Q]["GET"] =
            no_throw([this](shared_ptr<HttpServer::Response> res, shared_ptr<HttpServer::Request> req) {
                rfc3986::Form query(http_query(req->path));
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
                float anno_factor = query.get<float>("anno_factor", 0);
                        LOG(INFO) << "ANNO: " << anno_factor;
                ImageLoader loader(conf);
                ImageLoader::PerturbVector pv;
                int id = query.get<int>("id", rng() % db.size());
                ImageLoader::Value v;
                loader.sample(rng, &pv);
                loader.load([this, id](Record *r){db.read(id, r);}, pv, &v);
                ImageEncoder encoder;
                string buf;
                cv::Mat image = v.image;
                if (conf.annotate.size()) {
                    image = v.annotation;
                    if (anno_factor) {
                        image *= anno_factor;
                    }
                }
                encoder.encode(image, &buf);
                *res << "HTTP/1.1 200 OK\r\n";
                *res << "Content-Type: image/jpeg\r\n";
                *res << "Content-Length: " << buf.size() << "\r\n\r\n";
                *res << buf;
            });
        default_resource["GET"] =
            no_throw([this](shared_ptr<HttpServer::Response> res, shared_ptr<HttpServer::Request> req) {
                string path = req->path;
                if (path == "/") {
                    path = "/index.html";
                }
                auto it = statics.find(path);
                if (it != statics.end()) {
                    auto text = it->second;
                    *res << "HTTP/1.1 200 OK\r\n";
                    *res << "Content-Type: " << path2mime(path) << "\r\n";
                    *res << "Content-Length: " << (text.second - text.first) << "\r\n\r\n";
                    res->write(text.first, text.second - text.first);
                }
                else {
                    *res << "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
                }
            });
    }

    ~HttpServer () {
        magic_close(cookie);
    }

};

int main(int argc, char const* argv[]) {
    banner();

    //string address;
    unsigned short port;
    int threads;
    fs::path db_path;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        //("address", po::value(&address)->default_value("0.0.0.0"), "")
        ("port", po::value(&port)->default_value(18888), "")
        ("db", po::value(&db_path), "")
        ("threads,t", po::value(&threads)->default_value(1), "")
        ("no-browser", "do not start browser")
        ;
    po::options_description desc_hidden("Expert options");
    desc_hidden.add_options()
        //("html_root", po::value(&html_root), "")
        ;
    desc.add(desc_hidden);

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

    HttpServer server(db_path, port, threads);
    thread th([&]() {
                LOG(INFO) << "listening at port: " << port;
                LOG(INFO) << "running server with " << threads << " threads.";
                server.start();
            });
    do { // test and start web browser
        if (vm.count("no-browser")) break;
        char *display = getenv("DISPLAY");
        if ((!display) || (strlen(display) == 0)) {
            LOG(WARNING) << "No DISPLAY found, not starting browser.";
            break;
        }
        boost::format cmd("xdg-open http://localhost:%1%");
        system((cmd % port).str().c_str());
    } while (false);
    // GET /hello
    th.join();
    return 0;
}

