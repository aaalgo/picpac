#define BOOST_SPIRIT_NO_PREDEFINED_TERMINALS
#include <chrono>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <magic.h>
#include <json11.hpp>
#include <server_http.hpp>
#include <server_extra.hpp>
#include <server_extra_zip.hpp>
#include "picpac-cv.h"
#include "picpac-overview.h"
#include "bfdfs/bfdfs.h"

using namespace std;
using namespace picpac;

char const *version = PP_VERSION "-" PP_BUILD_NUMBER "," PP_BUILD_ID "," PP_BUILD_TIME;

extern char _binary_html_static_start;

static vector<pair<string, string>> const EXT_MIME{
    {".html", "text/html"},
    {".css", "text/css"},
    {".js", "application/javascript"},
};
static unsigned constexpr EXT_MIME_GZIP = 3;
static std::string const DEFAULT_MIME("application/octet-stream");

void banner () {
    cout << "PicPac Server" << endl;
    cout << "Version: " << version << endl;
    cout << "https://github.com/aaalgo/picpac" << endl;
}

using namespace json11;

static size_t max_peak_mb = 1000;
static float max_peak_relax = 0.2;

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;
using SimpleWeb::Response;
using SimpleWeb::Request;

class Service:  public SimpleWeb::Multiplexer {
    picpac::IndexedFileReader db;
    bfdfs::Loader statics;
    default_random_engine rng;  // needs protection!!!TODO
    string overview;
    string collage;
    vector<vector<string>> samples;
    magic_t cookie;

#define HTTP_Q "(\\?.+)?$"

    string http_query (string const &path) const {
        auto p = path.find('?');
        if (p == path.npos) return "";
        return path.substr(p+1);
    }


    static string const &path2mime (string const &path, bool *gzip) {
        do {
            auto p = path.rfind('.');
            if (p == path.npos) break;
            string ext = path.substr(p);
            *gzip = false;
            unsigned i = 0; 
            for (auto const &v: EXT_MIME) {
                if (v.first == ext) {
                    if (i < EXT_MIME_GZIP) {
                        *gzip = true;
                    }
                    return v.second;
                }
                ++i;
            }
        } while (false);
        return DEFAULT_MIME;
    }

public:
    Service (fs::path const &db_path, HttpServer *sv)
        : Multiplexer(sv),
        db(db_path),
        statics(&_binary_html_static_start) {

        cookie = magic_open(MAGIC_MIME_TYPE);
        CHECK(cookie);
        magic_load(cookie, "/usr/share/misc/magic:/usr/local/share/misc/magic");
        Overview ov(db, max_peak_mb, max_peak_relax);
        ov.toJson(&overview);
        {
            ImageEncoder encoder(".jpg");
            encoder.encode(ov.collage(), &collage);
            {
                vector<vector<cv::Mat>> ss;
                ov.stealSamples(&ss);
                samples.resize(ss.size());
                for (unsigned i = 0; i < ss.size(); ++i) {
                    auto &from = ss[i];
                    auto &to = samples[i];
                    to.resize(from.size());
                    for (unsigned j = 0; j < from.size(); ++j) {
                        encoder.encode(from[j], &to[j]);
                    }
                }
            }
        }

        add("^/api/overview" HTTP_Q, "GET", [this](Response &res, Request &req) {

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
                res.content = overview;
                res.mime = "application/json";
            });
        add("^/api/collage.jpg" HTTP_Q, "GET", [this](Response &res, Request &req) {

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
                res.content = collage;
                res.mime = "image/jpeg";
            });
        add("^/api/thumb" HTTP_Q, "GET", [this](Response &res, Request &req) {
                int cls = req.GET.get<int>("class", 0);
                int off = req.GET.get<int>("offset", 0);
                if (cls >= samples.size()
                        || off >= samples[cls].size()) {
                    res.status = 404;
                    res.status_string = "Not Found";
                }
                else {
                    res.mime = "image/jpeg";
                    res.content = samples[cls][off];
                }
            });
        add("^/api/sample" HTTP_Q, "GET", [this](Response &res, Request &req) {
                int count = req.GET.get<int>("count", 20);
                //string anno = query.get<string>("anno", "");
                vector<int> ids(count);
                for (auto &id: ids) {
                    id = rand() % db.size();
                }
                Json json = Json::object {
                    {"samples", Json(ids)}
                };
                res.mime = "application/json";
                res.content = json.dump();
            });
        add("^/api/file" HTTP_Q, "GET", 
            [this](Response &res, Request &req) {
                int id = req.GET.get<int>("id", 0);
                Record rec;
                db.read(id, &rec);
                string buf = rec.field_string(0);
                res.mime = magic_buffer(cookie, &buf[0], buf.size());
                res.content.swap(buf);
            });
        add("^/api/image" HTTP_Q, "GET", 
            [this](Response &res, Request &req) {
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
#define PICPAC_CONFIG_UPDATE(C,P) C.P = req.GET.get<decltype(C.P)>(#P, C.P)
                PICPAC_CONFIG_UPDATE_ALL(conf);
#undef PICPAC_CONFIG_UPDATE
                float anno_factor = req.GET.get<float>("anno_factor", 0);
                        LOG(INFO) << "ANNO: " << anno_factor;
                bool do_norm = req.GET.get<int>("norm", 0);
                ImageLoader loader(conf);
                ImageLoader::PerturbVector pv;
                int id = req.GET.get<int>("id", rng() % db.size());
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
                if (do_norm) {
                    cv::Mat tmp;
                    cv::normalize(image, tmp, 0, 255, cv::NORM_MINMAX, CV_8U);
                    image = tmp;
                }
                encoder.encode(image, &buf);
                res.mime = "image/jpeg";
                res.content.swap(buf);
            });
        add_default("GET",
            [this](Response &res, Request &req) {
                string path = req.path;
                auto it = statics.find(req.path);
                if (it == statics.end()) {
                    path = "/index.html";
                    it = statics.find(path);
                }
                if (it != statics.end()) {
                    auto page = it->second;
                    auto it = req.header.find("If-None-Match");
                    do {
                        if (it != req.header.end()) {
                            string const &tag = it->second;
                            if (tag == page.checksum || (true 
                                && (tag.size() == (page.checksum.size() + 2))
                                && (tag.find(page.checksum) == 1)
                                && (tag.front() == '"')
                                && (tag.back() == '"'))) {
                                res.status = 304;
                                res.status_string = "Not Modified";
                                res.content.clear();
                                break;
                            }
                        }
                        res.content = string(page.begin, page.end);
                        bool gz = false;
                        res.mime = path2mime(path, &gz);
                        res.header.insert(make_pair(string("ETag"), '"' + page.checksum +'"'));
                        if (gz) {
                            SimpleWeb::plugins::deflate(res, req);
                        }
                    } while (false);
                }
                else {
                    res.status = 404;
                    res.content.clear();
                }
            });
    }

    ~Service () {
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
        ("max-peek", po::value(&max_peak_mb)->default_value(1000), "read this number MB of data")
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

    HttpServer server(port, threads);
    Service service(db_path, &server);
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

