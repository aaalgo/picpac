#include <fstream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/raw_function.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include "picpac.h"
#include "picpac-cv.h"
using namespace boost::python;
using namespace picpac;

namespace {

template <typename T>
T *get_ndarray_data (object &o) {
    PyArrayObject *nd = reinterpret_cast<PyArrayObject *>(o.ptr());
    return reinterpret_cast<T*>(PyArray_DATA(nd));
}

size_t get_ndarray_nbytes (object &o) {
    PyArrayObject *nd = reinterpret_cast<PyArrayObject *>(o.ptr());
    return size_t(PyArray_NBYTES(nd));
}

class NumpyBatchImageStream: public BatchImageStream {
public:
    NumpyBatchImageStream (std::string const &path, Config const &c)
        : BatchImageStream(fs::path(path), c) {
    }
    tuple next () {
        vector<npy_intp> images_dims;
        vector<npy_intp> labels_dims;
        next_shape(&images_dims, &labels_dims);
        object images = object(boost::python::handle<>(PyArray_SimpleNew(images_dims.size(), &images_dims[0], NPY_FLOAT)));
        CHECK(images.ptr());
        float *images_buf = get_ndarray_data<float>(images);
        object labels = object(boost::python::handle<>(PyArray_SimpleNew(labels_dims.size(), &labels_dims[0], NPY_FLOAT)));
        CHECK(labels.ptr());
        float *labels_buf = get_ndarray_data<float>(labels);
        unsigned padding;
        next_fill(images_buf, labels_buf, &padding);
        return make_tuple(images, labels, padding);
    }
};

object create_image_stream (tuple args, dict kwargs) {
    object self = args[0];
    CHECK(len(args) > 1);
    string path = extract<string>(args[1]);
    NumpyBatchImageStream::Config config;
    /*
    bool train = extract<bool>(kwargs.get("train", true));
    unsigned K = extract<unsigned>(kwargs.get("K", 1));
    unsigned fold = extract<unsigned>(kwargs.get("fold", 0));
    if (K <= 1) {
        if (!train) {
            config.loop = false;
            config.reshuffle = false;
        }
    }
    else {
        config.kfold(K, fold, train);
    }
    */
#define PICPAC_CONFIG_UPDATE(C, P) \
    C.P = extract<decltype(C.P)>(kwargs.get(#P, C.P)) 
    PICPAC_CONFIG_UPDATE_ALL(config);
#undef PICPAC_CONFIG_UPDATE
    if (kwargs.has_key("channel_first")) {
        LOG(ERROR) << "channel_first is depreciated, use order=\"NHWC\"";
        CHECK(false);
    }
    return self.attr("__init__")(path, config);
};

object return_iterator (tuple args, dict kwargs) {
    object self = args[0];
    self.attr("reset")();
    return self;
};

class NumpyMultiImageStream: public MultiImageStream {
public:
    struct Config: public MultiImageStream::Config {
        float mean_color1;
        float mean_color2;
        float mean_color3;
        unsigned onehot;
        unsigned batch;
        bool pad;
        bool bgr2rgb;
        string order;
        Config ():
            mean_color1(0),
            mean_color2(0),
            mean_color3(0),
            onehot(0), batch(1), pad(false), bgr2rgb(false) {
        }
    };
    Config config;
    NumpyMultiImageStream (std::string const &path, Config const &c)
        : MultiImageStream(fs::path(path), c), config(c) {
    }
    tuple next () {
        CHECK(config.onehot == 0);
        CHECK(config.batch == 1);
        CHECK(config.pad == false);
        Value v(MultiImageStream::next());
        vector<npy_intp> images_dims;
        vector<npy_intp> labels_dims;
        CHECK(v.images.size() > 0);
        cv::Mat const &im0 = v.images[0];
        CHECK(im0.data);
        images_dims.push_back(v.images.size());
        images_dims.push_back(im0.rows);
        images_dims.push_back(im0.cols);
        images_dims.push_back(im0.channels());

        labels_dims.push_back(1);
        labels_dims.push_back(v.annotation.rows);
        labels_dims.push_back(v.annotation.cols);
        labels_dims.push_back(1);

        object images = object(boost::python::handle<>(PyArray_SimpleNew(images_dims.size(), &images_dims[0], NPY_FLOAT)));
        CHECK(images.ptr());
        float *images_buf = get_ndarray_data<float>(images);
        float *images_buf0 = images_buf;
        // copy images
        cv::Scalar mean0{config.mean_color1, config.mean_color2, config.mean_color3};
        for (unsigned i = 0; i < v.images.size(); ++i) {
            cv::Mat const &im = v.images[i];
            CHECK(im.rows == im0.rows);
            CHECK(im.cols == im0.cols);
            CHECK(im.channels() == im0.channels());
            images_buf = impl::copy<float>(im, images_buf, mean0, config.bgr2rgb);
        }
        CHECK((images_buf - images_buf0) * sizeof(float) == get_ndarray_nbytes(images));
        object labels = object(boost::python::handle<>(PyArray_SimpleNew(labels_dims.size(), &labels_dims[0], NPY_FLOAT)));
        CHECK(labels.ptr());
        float *labels_buf = get_ndarray_data<float>(labels);
        float *labels_buf0 = labels_buf;
        labels_buf = impl::copy<float>(v.annotation, labels_buf, mean0, config.bgr2rgb);
        CHECK((labels_buf - labels_buf0) * sizeof(float) == get_ndarray_nbytes(labels));
        return make_tuple(v.label, images, labels);
    }
};

object create_multi_image_stream (tuple args, dict kwargs) {
    object self = args[0];
    CHECK(len(args) > 1);
    string path = extract<string>(args[1]);
    NumpyMultiImageStream::Config config;
    /*
    bool train = extract<bool>(kwargs.get("train", true));
    unsigned K = extract<unsigned>(kwargs.get("K", 1));
    unsigned fold = extract<unsigned>(kwargs.get("fold", 0));
    if (K <= 1) {
        if (!train) {
            config.loop = false;
            config.reshuffle = false;
        }
    }
    else {
        config.kfold(K, fold, train);
    }
    */
#define PICPAC_CONFIG_UPDATE(C, P) \
    C.P = extract<decltype(C.P)>(kwargs.get(#P, C.P)) 
    PICPAC_CONFIG_UPDATE_ALL(config);
#undef PICPAC_CONFIG_UPDATE
    config.cache = 0;
    return self.attr("__init__")(path, config);
};

class Writer: public FileWriter {
    int nextid;
public:
    Writer (string const &path): FileWriter(fs::path(path), FileWriter::COMPACT), nextid(0) {
    }

    void setNextId (int v) {
        nextid = v;
    }

    void append (float label, string const &buf) {
        Record record(label, buf);
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }

    void append (string const &buf1, string const &buf2) {
        Record record(0, buf1, buf2);
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }

    void append (float label, string const &buf1, string const &buf2) {
        Record record(label, buf1, buf2);
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }

    void append (float label, string const &buf1, string const &buf2, string const &buf3) {
        Record record(label, buf1, buf2, buf3);
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }

    void append (float label, string const &buf1, string const &buf2, string const &buf3, string const &buf4) {
        Record record(label, buf1, buf2, buf3, buf4);
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }

    void append (float label, string const &buf1, string const &buf2, string const &buf3, string const &buf4, string const &buf5) {
        Record record(label, buf1, buf2, buf3, buf4, buf5);
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }
};

class Reader: public IndexedFileReader {
    int _next;
    object ctor;
public:
    Reader (string const &path): IndexedFileReader(path), _next(0) {
        auto collections = import("collections");
        auto namedtuple = collections.attr("namedtuple");
        list fields;
        fields.append("id"); 
        fields.append("label"); 
        fields.append("label2");
        fields.append("fields");
        ctor = namedtuple("Record", fields);
    }
    object next () {
        if (_next >= size()) {
            throw EoS();
        }
        return read(_next++);
    }
    void reset () {
        _next = 0;
    }
    object read (int i) {
        Record rec;
        IndexedFileReader::read(i, &rec);
        list fields;
        for (unsigned i = 0; i < rec.size(); ++i) {
            fields.append(rec.field_string(i));
        }
        auto const &meta = rec.meta();
        return ctor(meta.id, meta.label, meta.label2, fields);
    }
};

void serialize_raw_ndarray (object &obj, std::ostream &os) {
    PyArrayObject *image = reinterpret_cast<PyArrayObject *>(obj.ptr());
    int nd = PyArray_NDIM(image);
    CHECK(nd == 2 || nd == 3);
    auto desc = PyArray_DESCR(image);
    CHECK(desc);
    CHECK(PyArray_EquivByteorders(desc->byteorder, NPY_NATIVE)
            || desc->byteorder == '|') << "Only support native/little endian";
    int elemSize = desc->elsize;
    CHECK(elemSize > 0) << "Flex type not supported.";
    int ch = (nd == 2) ? 1 : PyArray_DIM(image, 2); 
    elemSize *= ch; // opencv elements includes all channels
    //CHECK(image->strides[1] == elemSize) << "Image cols must be consecutive";
    int rows = PyArray_DIM(image, 0);
    int cols = PyArray_DIM(image, 1);
    int t = PyArray_TYPE(image);
    int type = 0;
    switch (t) {
        case NPY_UINT8: type = CV_MAKETYPE(CV_8U, ch); break;
        case NPY_INT8: type = CV_MAKETYPE(CV_8S, ch); break;
        case NPY_UINT16: type = CV_MAKETYPE(CV_16U, ch); break;
        case NPY_INT16: type = CV_MAKETYPE(CV_16S, ch); break;
        case NPY_INT32: type = CV_MAKETYPE(CV_32S, ch); break;
        case NPY_FLOAT32: type = CV_MAKETYPE(CV_32F, ch); break;
        case NPY_FLOAT64: type = CV_MAKETYPE(CV_64F, ch); break;
        default: CHECK(0) << "type not supported: " << t;
    }
    int stride = PyArray_STRIDE(image, 0);
    CHECK(stride == cols * elemSize) << "bad stride";
    os.write(reinterpret_cast<char const *>(&type), sizeof(type));
    os.write(reinterpret_cast<char const *>(&rows), sizeof(rows));
    os.write(reinterpret_cast<char const *>(&cols), sizeof(cols));
    os.write(reinterpret_cast<char const *>(&elemSize), sizeof(elemSize));
    char const *off = PyArray_BYTES(image);
    for (int i = 0; i < rows; ++i) {
        os.write(off, cols * elemSize);
        off += stride;
    }
}

string encode_raw_ndarray (object &obj) {
    std::ostringstream ss;
    serialize_raw_ndarray(obj, ss);
    return ss.str();
}

void write_raw_ndarray (string const &path, object &obj) {
    std::ofstream os(path.c_str(), std::ios::binary);
    serialize_raw_ndarray(obj, os);
}

void (Writer::*append1) (float, string const &) = &Writer::append;
void (Writer::*append2) (string const &, string const &) = &Writer::append;
void (Writer::*append3) (float, string const &, string const &) = &Writer::append;

void (Writer::*append4) (float, string const &, string const &, string const &) = &Writer::append;

void (Writer::*append5) (float, string const &, string const &, string const &, string const &) = &Writer::append;

void (Writer::*append6) (float, string const &, string const &, string const &, string const &, string const &) = &Writer::append;

void translate_eos (EoS const &)
{
    // Use the Python 'C' API to set up an exception object
    PyErr_SetNone(PyExc_StopIteration);
}

}

BOOST_PYTHON_MODULE(_picpac)
{
    scope().attr("__doc__") = "PicPoc Python API";
    register_exception_translator<EoS>(&translate_eos);
    class_<NumpyBatchImageStream::Config>("ImageStreamParams", init<>());
    class_<NumpyBatchImageStream, boost::noncopyable>("ImageStream", no_init)
        .def("__init__", raw_function(create_image_stream), "exposed ctor")
        .def("__iter__", raw_function(return_iterator))
        .def(init<string, NumpyBatchImageStream::Config const&>()) // C++ constructor not exposed
        .def("next", &NumpyBatchImageStream::next)
        .def("size", &NumpyBatchImageStream::size)
        .def("reset", &NumpyBatchImageStream::reset)
        .def("categories", &NumpyBatchImageStream::categories)
    ;
    class_<NumpyMultiImageStream::Config>("MultiImageStreamParams", init<>());
    class_<NumpyMultiImageStream, boost::noncopyable>("MultiImageStream", no_init)
        .def("__init__", raw_function(create_multi_image_stream), "exposed ctor")
        .def("__iter__", raw_function(return_iterator))
        .def(init<string, NumpyMultiImageStream::Config const&>()) // C++ constructor not exposed
        .def("next", &NumpyMultiImageStream::next)
        .def("size", &NumpyMultiImageStream::size)
        .def("reset", &NumpyMultiImageStream::reset)
    ;
    class_<Reader>("Reader", init<string>())
        .def("__iter__", raw_function(return_iterator))
        .def("next", &Reader::next)
        .def("size", &Reader::size)
        .def("read", &Reader::read)
        .def("reset", &Reader::reset)
    ;
    class_<Writer>("Writer", init<string>())
        .def("append", append1)
        .def("append", append2)
        .def("append", append3)
        .def("append", append4)
        .def("append", append5)
        .def("append", append6)
        .def("setNextId", &Writer::setNextId);
    ;
    def("encode_raw", ::encode_raw_ndarray);
    def("write_raw", ::write_raw_ndarray);
#undef NUMPY_IMPORT_ARRAY_RETVAL
#define NUMPY_IMPORT_ARRAY_RETVAL
    import_array();
}

