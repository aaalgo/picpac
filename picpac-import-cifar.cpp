#include <fstream>
#include "picpac.h"
#include "picpac-cv.h"
#include "picpac-util.h"

using namespace std;
using namespace picpac;

static size_t constexpr WIDTH = 32;
static size_t constexpr HEIGHT = 32;
static size_t constexpr AREA = WIDTH * HEIGHT;
static size_t constexpr CHANNELS = 3;
static size_t constexpr DIM = WIDTH * HEIGHT * CHANNELS;

void load (string const &path, FileWriter &writer) {
    ImageEncoder encoder(".png");
    ifstream is(path.c_str(), ios::binary);
    size_t sz = 1 + DIM;
    is.seekg(0, ios::end);
    size_t total = is.tellg();
    is.seekg(0, ios::beg);
    BOOST_VERIFY(total % sz == 0);
    total /= sz;
    //cv::Mat image(HEIGHT, WIDTH, 
    vector<cv::Mat> channels(3);
    for (auto &m: channels) {
        m.create(HEIGHT, WIDTH, CV_8UC1);
    }
    cv::Mat image;
    string buf;
    for (unsigned i = 0; i < total; ++i) {
        uint8_t label;
        is.read((char *)&label, sizeof(label));
        is.read(channels[2].ptr<char>(0), AREA);
        is.read(channels[1].ptr<char>(0), AREA);
        is.read(channels[0].ptr<char>(0), AREA);
        CHECK(is);
        cv::merge(channels, image);
        encoder.encode(image, &buf);
        picpac::Record rec((int)label, buf);
        writer.append(rec);
    }
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    {
        picpac::FileWriter writer("train.db", FileWriter::COMPACT);
        load("cifar-10-batches-bin/data_batch_1.bin", writer);
        load("cifar-10-batches-bin/data_batch_2.bin", writer);
        load("cifar-10-batches-bin/data_batch_3.bin", writer);
        load("cifar-10-batches-bin/data_batch_4.bin", writer);
        load("cifar-10-batches-bin/data_batch_5.bin", writer);
    }
    {
        picpac::FileWriter writer("test.db", FileWriter::COMPACT);
        load("cifar-10-batches-bin/test_batch.bin", writer);
    }

    return 0;
}


