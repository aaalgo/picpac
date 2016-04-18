#include <fstream>
#include <boost/program_options.hpp>
#include "picpac-cv.h"


/**
 * This program extract positive regions from the input db
 * and dump the vec file for opencv cascade training, assuming
 * - Each training image has exactly one positive region.
 * The program first computes the average aspect ratio, and
 * the set the width and height for positive region using the
 * aspect ratio and the given "min" side width.
 * The resulting width and height printed should be used for
 * invoking opencv_traincascade.
 */

using namespace std;
using namespace picpac;

cv::Rect bound (cv::Mat image) {
    int minx = image.cols, maxx = 0;
    int miny = image.rows, maxy = 0;
    for (int y = 0; y < image.rows; ++y) {
        uint8_t const *line = image.ptr<uint8_t const>(y);
        for (int x = 0; x < image.cols; ++x) {
            if (line[x] > 0) {
                if (x < minx) minx = x;
                if (x > maxx) maxx = x;
                if (y < miny) miny = y;
                if (y > maxy) maxy = y;
            }
        }
    }
    CHECK(minx < maxx);
    CHECK(miny < maxy);
    return cv::Rect(minx, miny, maxx-minx+1, maxy-miny+1);
}

int main(int argc, char const* argv[]) {
    ImageStream::Config config;
    config.loop = false;
    config.stratify = false;
    config.anno_type = CV_8U;
    config.anno_color1 = 1;
    config.anno_color2 = 0;
    config.anno_color3 = 0;
    config.anno_copy = false;
    config.anno_thickness = -1;

    fs::path db_path;
    fs::path vec_path;
    fs::path dir_path;
    int min;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("annotate,a", po::value(&config.annotate)->default_value("json"), "")
        ("db", po::value(&db_path), "")
        ("vec", po::value(&vec_path), "")
        ("dir", po::value(&dir_path), "")
        ("min", po::value(&min)->default_value(32), "")
        ;

    po::positional_options_description p;
    p.add("db", 1);
    p.add("vec", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help") || db_path.empty() || vec_path.empty()) {
        cout << "Usage:" << endl;
        cout << "\tpicpac-stat ... <db>" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }
    ImageStream db(db_path, config);
    // detect aspect ratio
    float ratio_sum = 0;
    int ratio_total = 0;
    for (;;) {
        try {
            ImageStream::Value v(db.next());
            cv::Rect bb = bound(v.annotation);
            ratio_sum += bb.width / bb.height;
            ratio_total += 1;


        }
        catch (EoS const &) {
            break;
        }
    }
    float ratio = ratio_sum / ratio_total;
    int width, height;
    if (ratio < 1) {
        width = min;
        height = min / ratio;
    }
    else {
        height = min;
        width = min * ratio;
    }
    cout << "width: " << width << endl;
    cout << "height: " << height << endl;

    db.reset();
    int c = 0;
    ofstream os(vec_path.c_str(), ios::binary);

    int vecsize = width * height;
    short tmp = 0;
    os.write(reinterpret_cast<char const *>(&ratio_total), sizeof(ratio_total));
    os.write(reinterpret_cast<char const *>(&vecsize), sizeof(vecsize));
    os.write(reinterpret_cast<char const *>(&tmp), sizeof(tmp));
    os.write(reinterpret_cast<char const *>(&tmp), sizeof(tmp));

    if (!dir_path.empty()) {
        fs::create_directories(dir_path);
    }
    for (;;) {
        try {
            ImageStream::Value v(db.next());
            if (v.image.channels() == 3) {
                cv::Mat tmp;
                cv::cvtColor(v.image, tmp, CV_BGR2GRAY);
                v.image = tmp;
            }
            cv::Rect bb = bound(v.annotation);
            CHECK(v.image.type() == CV_8U);
            cv::Mat roi = v.image(bb);
            cv::Mat sample;
            cv::resize(roi, sample, cv::Size(width, height));
            CHECK(sample.total() == width * height);
            short tmp = 0;
            os.write(reinterpret_cast<char const *>(&tmp), sizeof(tmp));
            for (int y = 0; y < sample.rows; ++y) {
                uint8_t const *line = sample.ptr<uint8_t const>(y);
                for (int x = 0; x < sample.cols; ++x) {
                    tmp = line[x];
                    os.write(reinterpret_cast<char const *>(&tmp), sizeof(tmp));

                }
            }
            if (!dir_path.empty()) {
                fs::path o = dir_path / (lexical_cast<string>(c) + ".jpg");
                cv::imwrite(o.native(), sample);
            }
            ++c;
        }
        catch (EoS const &) {
            break;
        }
    }
    CHECK(c == ratio_total);
    os.seekp(0);
    os.write(reinterpret_cast<char const *>(&c), sizeof(c));
    cout << c << " samples loaded." << endl;

    return 0;
}

