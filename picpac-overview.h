#include <vector>
#include <map>
#include <mutex>
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
#include <json11.hpp>
#include "picpac-cv.h"

namespace picpac {

namespace ba = boost::accumulators;
typedef ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::min, ba::tag::max, ba::tag::count, ba::tag::variance, ba::tag::moment<2>>> Stat;
using namespace json11;
// This class wrapes the accumulators for the features of images we are interested in
// Currently it only count the mean image area in pixel and mean height-width-ratio
class ImageStat {
    static size_t constexpr THUMBNAIL_SIZE = 200;
    Stat scale_stat;
    Stat height_width_ratio_stat;
    std::map<float, std::vector<cv::Mat>> groups;

    cv::Mat gen_thumb (cv::Mat image) const {
        cv::Mat thumb = image;
        {
            cv::Mat tmp;
            LimitSize(thumb, 0, THUMBNAIL_SIZE, &tmp);
            thumb = tmp;
        }
        if (thumb.channels() == 1) {
            cv::Mat tmp;
            cv::cvtColor(thumb, tmp, CV_GRAY2BGR);
            thumb = tmp;
        }
        if (thumb.type() != CV_8UC3) {
            cv::Mat tmp;
            cv::normalize(thumb, tmp, 0, 255, cv::NORM_MINMAX);
            thumb = tmp;
        }
        return thumb;
    }

    std::mutex mutex;
public:
    ImageStat () {
    }
    void addImage(cv::Mat &image, float label) {
        cv::Mat thumb = gen_thumb(image);
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (image.rows<=0 || image.cols<=0)
                return;
            double scale = std::sqrt(image.rows*image.cols);
            double height_width_ratio = (image.rows+0.0f)/image.cols;
            if (scale > 0) {
                scale_stat(scale);
            }
            if (height_width_ratio > 0) {
                height_width_ratio_stat(height_width_ratio);
            }
            groups[label].emplace_back(thumb);
        }
    }
    double meanScale() const {
        return ba::extract_result<ba::tag::mean>(scale_stat);
    }
    double meanHWRatio() const {
        return ba::extract_result<ba::tag::mean>(height_width_ratio_stat);
    }

    void getSamples (vector<cv::Mat> *samples) {
        for (auto &p: groups) {
            samples->insert(samples->end(), p.second.begin(),
                                          p.second.end());
        }
    }

    void getGroups (vector<vector<cv::Mat>> *samples) {
        int n = 0;
        for (auto const &p: groups) {
            int l = int(p.first);
            if (l > n) {
                n = l;
            }
        }
        n += 1;
        samples->clear();
        samples->resize(n);
        for (auto const &p: groups) {
            int l = int(p.first);
            auto &v = samples->at(l);
            v.insert(v.end(), p.second.begin(), p.second.end());
        }
    }

    cv::Mat collage (int size = 640) {
        cv::Mat all(size, size, CV_8UC3, cv::Scalar(0xFF, 0xFF, 0xFF));
        vector<cv::Mat> samples;
        getSamples(&samples);
        std::random_shuffle(samples.begin(), samples.end());
        if (samples.size() > 200) samples.resize(200);
        unsigned n = samples.size();
        if (n == 0) n = 1;
        int onesz = size / sqrt(1.0 * n) * 2;
        // n = 1:  onesz = size
        if (onesz > THUMBNAIL_SIZE) onesz = THUMBNAIL_SIZE;
        if (onesz < 32) onesz = 32;
        if (onesz >= size) onesz = size - 1;
        int cc = 0;
        int nxn = (size + onesz - 1) / onesz;
        for (auto &m: samples) {
            cv::Mat xx = m;
            {
                cv::Mat tmp;
                LimitSize(xx, 0, onesz, &tmp);
                xx = tmp;
            }
            int y = cc / nxn;
            int x = cc % nxn;
            if (y >= nxn) {
                x = rand() % (size - xx.cols);
                y = rand() % (size - xx.rows);
            }
            else {
                y *= onesz;
                x *= onesz;
                y = std::min(y, size - xx.rows);
                x = std::min(x, size - xx.cols);
            }
            xx.copyTo(all(cv::Rect(x, y, xx.cols, xx.rows)));
            cc += 1;
        }
        return all;
    }
};

// The class cans an image dataset and produces all kinds of
// statistics.  It also measures how fast we can read records.
// A PicPac database is designed such that upon opening,
// the location information of all images are loaded into memory, but not
// the image records themselves.
// struct Locator {      // see picpac.h
//      off_t offset;    // offset within file
//      uint32_t size;   // size of record
//      float group;     // see below    // see below
//      ......
// }
// We can therefore produce precise
// statistics on locator & group based on the locators.
// After that, we randomly sample at most "max_peak_mb" megabytes (1G by default)
// of records.  We load these records into memory.
// Each record has some meta data:
// struct Meta {        // see picpac.h
//      float label;
//      uint8_t width;  // number of fields.  The first field (index 0) is the image.
//                      // The second field, if present, is the annotation.  Others are not used.
//
//      int16_t label2; // optional secondary label, for stratification
//      ......
// }
// Each record has a label (or label1) and label2.  Locator::group is always
// the value of either label or label2, which is determined upon the creation
// of database.  Label is float, though for classification problems it should
// be of integral values.  Label2 is always integer.
//
class Overview {
    static size_t constexpr MB = 1024 * 1024;
    int total_cnt;      // total number of records
    int sample_cnt;     // number of sampled records
    size_t total_size;  // size in bytes
    size_t sample_size;
    // precise stat
    Stat size_stat;     // per-record size statistics
    std::map<int, int> group_cnt;
                        // # of different group values
    // estimation
    int group_is_float_cnt;
    int label1_is_float_cnt;
    int label2_is_float_cnt;
    int group_isnt_label1_cnt;  // either this
    int group_isnt_label2_cnt;  // or this should be 0
    int annotation_cnt;
    std::map<float, int> label1_cnt;
    std::map<int16_t, int> label2_cnt;
    std::map<string, int> mime_cnt;  // not implemented
    std::map<int, int> width_cnt;
    std::map<string, int> shape_cnt;


    float scan_time;

    ImageStat image_stat;
    vector<vector<cv::Mat>> samples;

    static bool is_float (float v) {
        return float(int(v)) != v;
    }

    template <typename T>
    Json::array cnt_to_json (std::map<T, int> const &mm) {
        std::vector<std::pair<T, int>> cnts(mm.begin(), mm.end());
        std::sort(cnts.begin(), cnts.end());
        Json::array counts;
        for (auto const &p: cnts) {
            counts.push_back(Json::array{p.first, p.second});
        }
        return counts;
    }

public:
    Overview (picpac::IndexedFileReader &db, size_t max_peak_mb = 1000, float max_peak_relax = 0.2)
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
        // only access locator information, for speed.
        db.loopIndex([this](Locator const &l) {
            total_size += l.size;
            size_stat(l.size);
            if (is_float(l.group)) ++group_is_float_cnt;
            group_cnt[int(l.group)] += 1;
        });
        if (total_size < max_peak * (1.0 + max_peak_relax)) {
            max_peak = total_size;
        }
        // Random sample records of given total size.
        std::vector<unsigned> ids(db.size());
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
#pragma omp parallel for
            for (unsigned id_id = 0; id_id < ids.size(); ++id_id) {
                unsigned id = ids[id_id];
                Record rec;
                Locator const &loc = db.locator(id);
                db.read(id, &rec);
                float label1 = rec.meta().label;
                float label2 = rec.meta().label2;
#pragma omp critical
                {
                    // rec now holds the record.
                    // we can use:  (details see class Record in picpac.h)
                    //  rec.meta()  // meta data
                    //  rec.field() or rec.field_string() to access the filed
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
                }
                cv::Mat image = decode_buffer(rec.field(0), -1);
                image_stat.addImage(image, label1);
#pragma omp critical
                ++progress;
            }
            scan_time = timer.elapsed().wall/1e9;
        }
        bool is_classification = (label1_is_float_cnt == 0) && (label1_cnt.size() > 1);
        if (is_classification) {
            image_stat.getGroups(&samples);
        }
        else {
            samples.clear();
            samples.resize(1);
            image_stat.getSamples(&(samples.front()));
        }
        std::cerr << "Scanned " << (1.0 * sample_size / MB) << "MB of data in " << scan_time << " seconds." << std::endl;
    }
private:
    template <typename T>
    Json field (string const &key, string const &name, T const &value, bool display = true) {
        return Json::object{{"key", key}, {"name", name}, {"value", value}, {"display", display}};
    }
public:
    cv::Mat collage () {
        return image_stat.collage();
    }

    void stealSamples (vector<vector<cv::Mat>> *samples) {
        samples->swap(this->samples);
    }

    std::string guessTask () {
        if (shape_cnt.size()) return "segmentation";
        if (label1_is_float_cnt > 0 && label1_cnt.size() > 1) return "regression";
        if (label1_cnt.size() > 1) return "classification";
        return "none";
    }

    void toJson (string *buf) {
        bool all_scanned = sample_cnt >= total_cnt;
        Json::object all;
        Json::array summary{
            field("TOTAL_IMAGES", "Number of images", int(total_cnt)),
            field("TOTAL_MB", "Total megabytes", float(1.0 * total_size / MB)),
            field("GROUP_CNT", "Stratification groups", int(group_cnt.size()), group_cnt.size() > 1),
            field("GROUP_BY", "Stratification variable", 
                    (group_isnt_label1_cnt == 0) ? 
                          (group_is_float_cnt == 0 ? "label" : "int(label)")
                        : ((group_isnt_label2_cnt == 0) ?
                                (group_is_float_cnt == 0 ? "label2" : "int(label2)") : "NONE"),
                    group_cnt.size() > 1),
            field("GROUP_FLOAT_CNT", "", group_is_float_cnt, false),
            field("ALL_SCANNED", "All images scanned", all_scanned, false),
            field("SCANNED_IMAGES", "Scanned images", int(sample_cnt), !all_scanned),
            field("SCANNED_MB", "Scanned megabytes", float(1.0 * sample_size/MB), !all_scanned),
            field("SCAN_RATIO", "Scan ratio", float(1.0 * sample_cnt / total_cnt)),
            field("SCAN_TIME", "Scan seconds", scan_time),
            field("SCAN_THROUGHPUT", "Scan throughput", 1.0 * sample_cnt / scan_time),
            field("MEAN_SCALE", "Mean image scale [sqrt(WxH)]", float(image_stat.meanScale())),
            field("MEAN_HWR", "Mean height/width", float(image_stat.meanHWRatio())),
            field("TYPICAL_LAYOUT", "Typical layout",
                            image_stat.meanHWRatio() > 1? "portrait" : "landscape"),
            field("SHAPE_CNT", "Distinct annotation shapes", int(shape_cnt.size()), shape_cnt.size()>0),
            field("LABEL1_CNT", "Distinct label values", int(label1_cnt.size()), label1_is_float_cnt == 0),
            field("LABEL1_FLOAT_CNT", "", label1_is_float_cnt, false),
            field("LABEL2_CNT", "Distinct label2 values", int(label2_cnt.size()), label2_is_float_cnt == 0 && label2_cnt.size() > 1),
            field("LABEL2_FLOAT_CNT", "", label2_is_float_cnt, false),
            field("GROUP_ISNT_LABEL1", "", group_isnt_label1_cnt, false),
            field("GROUP_ISNT_LABEL2", "", group_isnt_label2_cnt, false),
            field("TASK", "Task", guessTask())
        };
        if (label1_cnt.size() == 1) {
            summary.push_back(field("LABEL1_VALUE", "Label value", label1_cnt.begin()->first, true));
        }
        else if (label1_cnt.size() > 1 && label1_is_float_cnt == 0) {
            all["LABEL1_CNT"] = cnt_to_json(label1_cnt);
        }
        if (label2_cnt.size() == 1) {
            float v = label2_cnt.begin()->first;
            summary.push_back(field("LABEL2_VALUE", "Label2 value", v, v != 0));
        }
        else if (label2_cnt.size() > 1 && label2_is_float_cnt == 0) {
            all["LABEL2_CNT"] = cnt_to_json(label2_cnt);
        }
        if (shape_cnt.size() == 1) {
            summary.push_back(field("SHAPE_TYPE", "Annotation shape", shape_cnt.begin()->first, true));
        }
        else if (shape_cnt.size() > 1) {
            all["SHAPE_CNT"] = cnt_to_json(shape_cnt);
        }
        all["SUMMARY"] = summary;
        {
            vector<int> sz;
            for (auto const &v: samples) { sz.push_back(v.size()); }
            all["SAMPLE_SIZES"] = Json(sz);
        }
        *buf = Json(all).dump();
    }
};

}
