#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <fstream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/raw_function.hpp>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include <numpy/ndarrayobject.h>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include "picpac.h"
#include "picpac-image.h"
using namespace boost::python;
using namespace picpac;
namespace {

class PythonImageStream: public ImageStream {
public:
    PythonImageStream (std::string const &path, Config const &c)
        : ImageStream(fs::path(path), c) {
    }
    list next () {
        list l;
        Value v(ImageStream::next());
        l.append(v.label);
        for (auto &im: v.facets) {
            CHECK(im.annotation.empty());
            l.append(boost::python::handle<>(pbcvt::fromMatToNDArray(im.image)));
        }
        return l;
    }
};

object create_image_stream (tuple args, dict kwargs) {
    object self = args[0];
    boost::python::object simplejson = boost::python::import("simplejson");
    PythonImageStream::Config config;
    string path = extract<string>(kwargs.get("db"));
#define PICPAC_CONFIG_UPDATE(C, P) \
    C.P = extract<decltype(C.P)>(kwargs.get(#P, C.P)) 
    PICPAC_CONFIG_UPDATE_ALL(config);
#undef PICPAC_CONFIG_UPDATE
    object transforms = kwargs.get("transforms", list());
    config.transforms = extract<string>(simplejson.attr("dumps")(transforms));
    return self.attr("__init__")(path, config);
};

object return_iterator (tuple args, dict kwargs) {
    object self = args[0];
    self.attr("reset")();
    return self;
};

class Writer: public FileWriter {
    int nextid;
public:
    Writer (string const &path, bool overwrite): FileWriter(fs::path(path), FileWriter::COMPACT | (overwrite ? FileWriter::OVERWRITE : 0)), nextid(0) {
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

#if (PY_VERSION_HEX >= 0x03000000)

static void *init_numpy() {
#else
	static void init_numpy(){
#endif

	import_array();
	return NUMPY_IMPORT_ARRAY_RETVAL;
}   



BOOST_PYTHON_MODULE(picpac)
{
	init_numpy();
    scope().attr("__doc__") = "PicPoc Python API";
    to_python_converter<cv::Mat,
                     pbcvt::matToNDArrayBoostConverter>();

    pbcvt::matFromNDArrayBoostConverter();
    register_exception_translator<EoS>(&translate_eos);
    class_<PythonImageStream::Config>("ImageStreamParams", init<>());
    class_<PythonImageStream, boost::noncopyable>("ImageStream", no_init)
        .def("__init__", raw_function(create_image_stream), "exposed ctor")
        .def("__iter__", raw_function(return_iterator))
        .def(init<string, PythonImageStream::Config const&>()) // C++ constructor not exposed
#if (PY_VERSION_HEX >= 0x03000000)
        .def("__next__", &PythonImageStream::next)
#endif
        .def("next", &PythonImageStream::next)
        .def("size", &PythonImageStream::size)
        .def("reset", &PythonImageStream::reset)
        //.def("categories", &PythonImageStream::categories)
    ;
    class_<Reader>("Reader", init<string>())
        .def("__iter__", raw_function(return_iterator))
#if (PY_VERSION_HEX >= 0x03000000)
        .def("__next__", &Reader::next)
#endif
        .def("next", &Reader::next)
        .def("size", &Reader::size)
        .def("read", &Reader::read)
        .def("reset", &Reader::reset)
    ;
    class_<Writer>("Writer", init<string, bool>())
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
//#undef NUMPY_IMPORT_ARRAY_RETVAL
//#define NUMPY_IMPORT_ARRAY_RETVAL
}

