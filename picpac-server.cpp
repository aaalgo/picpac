#define BOOST_SPIRIT_NO_PREDEFINED_TERMINALS
#include <chrono>
#include <mutex>
#include <vector>
#include <map>
#include <unordered_map>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#define timer timer_for_boost_progress_t
#include <boost/progress.hpp>
#undef timer
#include <boost/timer/timer.hpp>
#include <magic.h>
#include <json11.hpp>
#include <server_http.hpp>
#include <server_extra.hpp>
#include <server_extra_zip.hpp>
#include "picpac-cv.h"
#include "bfdfs/bfdfs.h"

using namespace std;
using namespace picpac;
namespace ba = boost::accumulators;
typedef ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::min, ba::tag::max, ba::tag::count, ba::tag::variance, ba::tag::moment<2>>> Stat;

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

static size_t max_peak_mb = 200;
static float max_peak_relax = 0.2;
class Overview {
    static size_t constexpr MB = 1024 * 1024;
    int total_cnt;
    int sample_cnt;
    size_t total_size;
    size_t sample_size;
    // precise stat
    Stat size_stat;
    map<float, int> group_cnt;
    // estimation
    int group_is_float_cnt;
    int label1_is_float_cnt;
    int label2_is_float_cnt;
    int group_isnt_label1_cnt;
    int group_isnt_label2_cnt;
    int annotation_cnt;
    map<float, int> label1_cnt;
    map<float, int> label2_cnt;
    map<string, int> mime_cnt;
    map<int, int> width_cnt;
    map<string, int> shape_cnt;

    float scan_time;

    static bool is_float (float v) {
        return float(int(v)) != v;
    }

    template <typename T>
    Json::array cnt_to_json (map<T, int> const &mm) {
        vector<pair<T, int>> cnts(mm.begin(), mm.end());
        std::sort(cnts.begin(), cnts.end());
        Json::array counts;
        for (auto const &p: cnts) {
            counts.push_back(Json::array{p.first, p.second});
        }
        return counts;
    }

public:
    Overview (picpac::IndexedFileReader &db)
        : total_cnt(0), sample_cnt(0),
        total_size(0), sample_size(0),
        group_is_float_cnt(0),
        label1_is_float_cnt(0),
        label2_is_float_cnt(0),
        group_isnt_label1_cnt(0),
        group_isnt_label2_cnt(0),
        annotation_cnt(0) {

        size_t max_peak = max_peak_mb * MB;

        total_cnt = db.size();
        db.loopIndex([this](Locator const &l) {
            total_size += l.size;
            size_stat(l.size);
            if (is_float(l.group)) ++group_is_float_cnt;
            group_cnt[l.group] += 1;
        });
        if (total_size < max_peak * (1.0 + max_peak_relax)) {
            max_peak = total_size;
        }
        vector<unsigned> ids(db.size());
        for (unsigned i = 0; i < ids.size(); ++i) {
            ids[i] = i;
        }
        std::random_shuffle(ids.begin(), ids.end());
        // determine # images to scan
        for (unsigned id: ids) {
            if (sample_size >= max_peak) break;
            Locator const &loc = db.locator(id);
            ++sample_cnt;
            sample_size += loc.size;
        }
        ids.resize(sample_cnt);
        std::cerr << "Scanning " << sample_cnt << "/" << total_cnt << " images..." << std::endl;
        
        {
            boost::progress_display progress(ids.size(), std::cerr);
            boost::timer::cpu_timer timer;
            for (unsigned id: ids) {
                Locator const &loc = db.locator(id);
                Record rec;
                db.read(id, &rec);
                float label1 = rec.meta().label;
                float label2 = rec.meta().label2;
                if (is_float(label1)) {
                    ++label1_is_float_cnt;
                }
                if (is_float(label2)) {
                    ++label2_is_float_cnt;
                }
                if (loc.group != label1) {
                    ++group_isnt_label1_cnt;
                }
                if (loc.group != label2) {
                    ++group_isnt_label2_cnt;
                }
                label1_cnt[label1] += 1;
                label2_cnt[label2] += 1;
                width_cnt[rec.meta().width] += 1;
                // TODO: mime
                if (rec.meta().width > 1) {
                    auto anno = rec.field_string(1);
                    auto tt = anno.find("\"type\"");
                    if (tt != anno.npos
                            && anno.find("geometry") != anno.npos){
                        auto b = anno.find('"', tt + 6);
                        if (b != anno.npos) {
                            auto e = anno.find('"', b + 1);
                            if (e != anno.npos) {
                                string shape = anno.substr(b+1, e - b- 1);
                                annotation_cnt += 1;
                                shape_cnt[shape] += 1;
                            }
                        }
                    }
                }
                ++progress;
            }
            scan_time = timer.elapsed().wall/1e9;
        }
        std::cerr << "Scanned " << (1.0 * sample_size / MB) << "MB of data in " << scan_time << " seconds." << std::endl;
    }

    void toJson (string *buf) {
        Json::object all;
        Json::array summary{
            Json::object{{"key", "Total images"}, {"value", int(total_cnt)}},
            Json::object{{"key", "Total size/MB"}, {"value", float(1.0 * total_size / MB)}},
            Json::object{{"key", "Group count"}, {"value", int(group_cnt.size())}},
            Json::object{{"key", "Group is float"}, {"value", group_is_float_cnt}},
            Json::object{{"key", "Scanned images"}, {"value", int(sample_cnt)}},
            Json::object{{"key", "Scanned size/MB"}, {"value", float(1.0 * sample_size/MB)}},
            Json::object{{"key", "All scanned"}, {"value", sample_cnt >= total_cnt}},
            Json::object{{"key", "Scan time/s"}, {"value", scan_time}},
            Json::object{{"key", "Label1 count"}, {"value", int(label1_cnt.size())}},
            Json::object{{"key", "label is float"}, {"value", label1_is_float_cnt}},
            Json::object{{"key", "Label2 count"}, {"value", int(label2_cnt.size())}},
            Json::object{{"key", "label2 is float"}, {"value", label2_is_float_cnt}},
            Json::object{{"key", "Group isnt label1"}, {"value", group_isnt_label1_cnt}},
            Json::object{{"key", "Group isnt label2"}, {"value", group_isnt_label2_cnt}},
            Json::object{{"key", "Shape count"}, {"value", int(shape_cnt.size())}},
        };
        if (group_cnt.size() == 1) {
            summary.push_back(Json::object{{"key", "Group value"}, {"value", group_cnt.begin()->first}});
        }
        else if (group_cnt.size() > 1 && group_is_float_cnt == 0) {
            all["group_cnt"] = cnt_to_json(group_cnt);
        }
        if (label1_cnt.size() == 1) {
            summary.push_back(Json::object{{"key", "Label1 value"}, {"value", label1_cnt.begin()->first}});
        }
        else if (label1_cnt.size() > 1 && label1_is_float_cnt == 0) {
            all["label1_cnt"] = cnt_to_json(label1_cnt);
        }
        if (label2_cnt.size() == 1) {
            summary.push_back(Json::object{{"key", "Label2 value"}, {"value", label2_cnt.begin()->first}});
        }
        else if (label2_cnt.size() > 1 && label2_is_float_cnt == 0) {
            all["label2_cnt"] = cnt_to_json(label2_cnt);
        }
        if (shape_cnt.size() == 1) {
            summary.push_back(Json::object{{"key", "Shape value"}, {"value", shape_cnt.begin()->first}});
        }
        else if (shape_cnt.size() > 1) {
            all["shape_cnt"] = cnt_to_json(shape_cnt);
        }
        all["summary"] = summary;
        *buf = Json(all).dump();
    }
};

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;
using SimpleWeb::Response;
using SimpleWeb::Request;

class Service:  public SimpleWeb::Multiplexer {
    picpac::IndexedFileReader db;
    bfdfs::Loader statics;
    default_random_engine rng;  // needs protection!!!TODO
    string overview;
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
        Overview ov(db);
        ov.toJson(&overview);

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
                    auto text = it->second;
                    res.content = string(text.first, text.second);
                    bool gz = false;
                    res.mime = path2mime(path, &gz);
                    if (gz) {
                        SimpleWeb::plugins::deflate(res, req);
                    }
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

