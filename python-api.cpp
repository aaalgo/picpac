#include <fstream>
#include "picpac.h"
#include "picpac-ts.h"
#undef NDEBUG
#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/xio.hpp>

using namespace picpac;

namespace {
    using std::unique_ptr;
    namespace py=pybind11;

class PyTimeSeriesStream: public TimeSeriesStream {
    int batch;
    py::object ctor;
public:
    struct Config: public TimeSeriesStream::Config {

        Config (py::dict const &kwargs) {
            py::object json_ = py::module_::import("json");

            //dict sampler = kwargs.get("sampler");
#define UPDATE_CONFIG(V, D) \
            if (D.contains(#V)) V = D[#V].cast<decltype(V)>()
            //UPDATE_CONFIG(seed, sampler);
            UPDATE_CONFIG(loop, kwargs);
            UPDATE_CONFIG(shuffle, kwargs);
            UPDATE_CONFIG(reshuffle, kwargs);
            UPDATE_CONFIG(stratify, kwargs);

            UPDATE_CONFIG(mixin, kwargs);
            UPDATE_CONFIG(mixin_group_reset, kwargs);
            UPDATE_CONFIG(mixin_group_delta, kwargs);
            UPDATE_CONFIG(mixin_max, kwargs);

            //dict loader = kwargs.get("loader");
            UPDATE_CONFIG(cache, kwargs);
            UPDATE_CONFIG(preload, kwargs);
            UPDATE_CONFIG(threads, kwargs);

            if (kwargs.contains("transforms")) {
                py::object tr = kwargs["transforms"];
                transforms = json_.attr("dumps")(tr).cast<string>();
            }

        }
    };

    PyTimeSeriesStream (py::dict kwargs) 
        : TimeSeriesStream(fs::path(kwargs["db"].cast<string>()), Config(kwargs)), batch(1) {
            UPDATE_CONFIG(batch, kwargs);

            py::object collections = py::module_::import("collections");
            py::object namedtuple = collections.attr("namedtuple");
            py::list fields;
            fields.append("id");       // np array
            fields.append("label");    // labels
            //fields.append("range");    // labels
            fields.append("json");
            ctor = namedtuple("Meta", fields);
    }

    py::list next () {
        // return a batch
        // if EOS, this will 

        xt::xtensor<int32_t, 1> ids;
        ids.resize({batch});
        xt::xtensor<float, 1> labels;
        labels.resize({batch});
        //xt::xtensor<float, 2> range;
        //range.resize({batch, 2});
        py::list meta_json;
        xt::xtensor<float, 2> time;     // batch x L
        xt::xtensor<float, 3> data;     // batch x L x C
        int L = -1, C = -1;
        // create batch, emplace first object
        for (int i = 0; i < batch; ++i) {
            // reset of the batch
            try {
                Value v(TimeSeriesStream::next());
                ids(i) = v.id;
                labels(i) = v.label;
                //range(i, 0) = v.begin;
                //range(i, 1) = v.end;
                meta_json.append(v.meta);

                if (i == 0) {
                    L = v.series[0].time.shape(0);
                    C = v.channels();
                    time.resize({batch, L});
                    data.resize({batch, L, C});
                }
                else {
                    CHECK(L == v.series[0].time.shape(0));
                    CHECK(C == v.channels());
                }
                int off = 0;
                for (auto const &s: v.series) {
                    CHECK(s.value.shape(0) == L);
                    for (int l = 0; l < L; ++l) {
                        if (off == 0) {
                            time(i, l) = s.time(l);
                        }
                        else {
                            CHECK(time(i, l) == s.time(l));
                        }
                        for (int c = 0; c < s.value.shape(1); ++c) {
                            data(i, l, off+c) = s.value(l, c);
                        }
                    }
                    off += s.value.shape(1);
                }
                CHECK(off == C);
            }
            catch (EoS const &) {
                if (i == 0) {
                    throw pybind11::stop_iteration();
                }
                else {
                    xt::xtensor<int32_t, 1> ids1(xt::view(ids, xt::range(0, i)));
                    xt::xtensor<float, 1> labels1(xt::view(labels, xt::range(0, i)));
                    //xt::xtensor<float, 2> range1(xt::view(range, xt::range(0, i), xt::all()));
                    xt::xtensor<float, 2> time1(xt::view(time, xt::range(0, i), xt::all()));
                    xt::xtensor<float, 3> data1(xt::view(data, xt::range(0, i), xt::all(), xt::all()));
                    ids = std::move(ids1);
                    labels = std::move(labels1);
                    //range = std::move(range1);
                    time = std::move(time1);
                    data = std::move(data1);
                }
                break;
            }
        }
        py::list r;
        r.append(ctor(ids, labels, meta_json));
        r.append(time);
        r.append(data);
        return r;
    }
};

py::object return_iterator (py::object self) {
    self.attr("reset")();
    return self;
}


py::tuple interp (xt::pytensor<float, 1> const &time,
                               xt::pytensor<float, 2> const &value,
                               float begin, int size, float step, bool extrapolate) {
    Series from, to;
    from.time = time;
    from.value = value;
    float max_gap = to.interp(from, begin, size, step, extrapolate);

    xt::pytensor<float, 2> v = to.value;
    
    return py::make_tuple(v, max_gap);
}



#if 0
class PyImageLoader {
    PyTimeSeriesStream::Config config;
    Transforms transforms;
    int dump;
    int dump_cnt;
    int decode_mode;
    bachelor::Order bachelor_order;
    bachelor::ColorSpace bachelor_colorspace;
    std::mutex mutex;
    random_engine rng;

    object __load_image (cv::Mat image) {
        //Py_BEGIN_ALLOW_THREADS
        int channels = config.channels;
        if (channels < 0) channels = image.channels();
        int type = CV_MAKETYPE(config.dtype, channels);

        if (image.type() != type) {
            cv::Mat tmp;
            image.convertTo(tmp, type);
            image = tmp;
        }

        Sample sample;
        sample.facets.emplace_back(image);

        // load object into sample
        // transform
        string pv;
        pv.resize(transforms.pv_size());
        {
            lock_guard guard(mutex);
            transforms.pv_sample(rng, &pv[0]);
        }
        transforms.apply(&sample, &pv[0]);
        //Py_END_ALLOW_THREADS

        NumpyBatch::Config conf;
        CHECK(sample.facets.size() == 1);
        auto &im = sample.facets[0];
        CHECK(im.type == Facet::IMAGE);
        conf.batch = 1;
        conf.height = im.image.rows;
        conf.width = im.image.cols;
        conf.channels = im.image.channels();
        conf.depth = im.image.depth();
        conf.order = bachelor_order;
        conf.colorspace = bachelor_colorspace;
        BachelorFacetData bb(conf);
        bb.fill_next(im.image);
        if (dump > 0 && dump_cnt < dump) {
            bb.dump("picpac_dump/" + lexical_cast<string>(dump_cnt));
            dump_cnt += 1;
        }
        return object(boost::python::handle<>(bb.detach()));
    }
public:
    PyImageLoader (dict kwargs): config(kwargs), transforms(json::parse(config.transforms)), dump(0), dump_cnt(0), rng(config.seed) {
        string order("NHWC");
        string colorspace("BGR");
        UPDATE_CONFIG(dump, kwargs);
        UPDATE_CONFIG(order, kwargs);
        UPDATE_CONFIG(colorspace, kwargs);
        if (order == "NHWC") {
            bachelor_order = bachelor::NHWC;
        }
        else if (order == "NCHW") {
            bachelor_order = bachelor::NCHW;
        }
        else CHECK(0) << "Unrecognized order: " << order;
        if (colorspace == "BGR") {
            bachelor_colorspace = bachelor::BGR;
        }
        else if (colorspace == "RGB") {
            bachelor_colorspace = bachelor::RGB;
        }
        else CHECK(0) << "Unrecognized colorspace: " << colorspace;

        decode_mode = cv::IMREAD_UNCHANGED;
        if (config.channels == 1) {
            decode_mode = cv::IMREAD_GRAYSCALE;
        }
        else if (config.channels == 3) {
            decode_mode = cv::IMREAD_COLOR;
        }
        if (dump) {
            fs::create_directory("picpac_dump");
            spdlog::warn("dumping image to picpac_dump/{batch}_{field}_{image}.png";
        }
    }

    object load_path (string const &path) {
        return __load_image(cv::imread(path, decode_mode));
    }

    object load_binary (PyObject *buf) {
        return __load_image(decode_buffer(pyobject2buffer(buf), decode_mode));
    }
};
#endif

template <typename T = float>
class SeriesBuffer {
    string buf;
public:
    SeriesBuffer (xt::pytensor<T,1> time,
                  xt::pytensor<T,2> value) {
        CHECK(time.shape(0) == value.shape(0));
        size_t time_sz = time.size() * sizeof(T);
        size_t value_sz = value.size() * sizeof(T);
        size_t sz = sizeof(uint32_t) * 2 + time_sz + value_sz;
        buf.resize(sz);
        char *off = &buf[0];
        *reinterpret_cast<uint32_t *>(off) = time.shape(0); off += sizeof(uint32_t);
        *reinterpret_cast<uint32_t *>(off) = value.shape(1); off += sizeof(uint32_t);
        //memcpy(off, time.data(), time_sz); off += time_sz;
        //memcpy(off, value.data(), value_sz); off += value_sz;
        for (int i = 0; i < time.shape(0); ++i) {
            *reinterpret_cast<T *>(off) = time(i); off += sizeof(T);
            for (int j = 0; j < value.shape(1); ++j) {
                *reinterpret_cast<T *>(off) = value(i, j); off += sizeof(T);
            }
        }
        CHECK(off - &buf[0] == buf.size());
    }

    const_buffer buffer () const {
        return const_buffer(&buf[0], buf.size());
    }

};

class Writer: public FileWriter {
    int nextid;
public:
    static int constexpr FLAG_OVERWRITE = OVERWRITE;
    Writer (string const &path, int flags): FileWriter(fs::path(path), flags), nextid(0) {
    }

    void setNextId (int v) {
        nextid = v;
    }

    void append (float label, string const &anno, xt::pytensor<float, 1> time, xt::pytensor<float, 2> value) {
        SeriesBuffer buf(time, value);
        Record record(label, const_buffer(&anno[0], anno.size()), buf.buffer());
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }

    void append (float label, string const &anno, xt::pytensor<float, 1> time, xt::pytensor<float, 2> value,
                xt::pytensor<float, 1> time2, xt::pytensor<float, 2> value2) {
        SeriesBuffer buf(time, value);
        SeriesBuffer buf2(time2, value2);
        Record record(label, const_buffer(&anno[0], anno.size()), buf.buffer(), buf2.buffer());
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }

};

#if 0
class Reader: public IndexedFileReader {
    unsigned _next;
    py::object ctor;
    py::object series_ctor;
public:
    Reader (string const &path): IndexedFileReader(path), _next(0) {
        auto collections = py::module_::import("collections");
        auto namedtuple = collections.attr("namedtuple");
        py::list fields;
        fields.append("id"); 
        fields.append("label"); 
        fields.append("label2");
        fields.append("annotation");
        fields.append("fields");
        ctor = namedtuple("Record", fields);
        py::list fields1;
        fields1.append("time");
        fields1.append("value");
        series_ctor = namedtuple("Series", fields1);
    }
    py::object next () {
        if (_next >= size()) {
            throw pybind11::stop_iteration();
        }
        return read(_next++);
    }
    void reset () {
        _next = 0;
    }
    py::object read (int i) {
        Record rec;
        IndexedFileReader::read(i, &rec);
        py::list fields;
        CHECK(rec.size() >= 1);
        string annotation;
        {
            const_buffer buf = rec.field(0);
            const char* b = boost::asio::buffer_cast<const char*>(buf);
            const char* e = b + boost::asio::buffer_size(buf);
            annotation = string(b, e);
        }


        for (unsigned i = 1; i < rec.size(); ++i) {
            Series series;
            series.decode(rec.field(i));
            xt::pytensor<float, 1> time = series.time;
            xt::pytensor<float, 2> value = series.value;
            fields.append(series_ctor(time, value));
        }
        auto const &meta = rec.meta();
        return ctor(meta.id, meta.label, meta.label2, annotation, fields);
    }
};
#endif

void (Writer::*append1) (float, string const &, xt::pytensor<float, 1>, xt::pytensor<float, 2>) = &Writer::append;
void (Writer::*append2) (float, string const &, xt::pytensor<float, 1>, xt::pytensor<float, 2>, xt::pytensor<float, 1>, xt::pytensor<float, 2>) = &Writer::append;
/*

void (Writer::*append4) (float, string const &, string const &, string const &) = &Writer::append;

void (Writer::*append5) (float, string const &, string const &, string const &, string const &) = &Writer::append;

void (Writer::*append6) (float, string const &, string const &, string const &, string const &, string const &) = &Writer::append;
*/

}


PYBIND11_MODULE(picpac_ts, module)
{
    using namespace picpac;
    xt::import_numpy();
    module.doc() = "PicPoc Python API";
    module.attr("OVERWRITE") = Writer::FLAG_OVERWRITE;

    module.def("interp", interp);

    py::class_<PyTimeSeriesStream>(module, "TimeSeriesStream")
        .def(py::init<py::dict>())
        .def("__iter__", return_iterator)
        .def("__next__", &PyTimeSeriesStream::next)
        .def("size", &PyTimeSeriesStream::size)
        .def("reset", &PyTimeSeriesStream::reset)
    ;
    /*
    class_<PyImageLoader, boost::noncopyable>("ImageLoader", init<dict>())
        .def("load_path", &PyImageLoader::load_path)
        .def("load_binary", &PyImageLoader::load_binary)
    ;
    */

#if 0
    py::class_<Reader>(module, "Reader")
        .def(py::init<string>())
        .def("__iter__", return_iterator)
        .def("__next__", &Reader::next)
        .def("next", &Reader::next)
        .def("size", &Reader::size)
        .def("read", &Reader::read)
        .def("reset", &Reader::reset)
    ;
#endif

    py::class_<Writer>(module, "Writer")
        .def(py::init<string, int>())
        .def("append", append1)
        .def("append", append2)
        .def("setNextId", &Writer::setNextId);
    ;

}


