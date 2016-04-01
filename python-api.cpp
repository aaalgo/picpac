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

class BatchImageStream: public ImageStream {
    unsigned onehot;
    unsigned batch;
    bool pad;
    static const bool rgb = false;

    template <typename Tfrom = uint8_t, typename Tto = float>
    Tto *split (cv::Mat image, Tto *buffer) {
        Tto *ptr_b = buffer;
        Tto *ptr_g = buffer;
        Tto *ptr_r = buffer;
        if (rgb) {
            ptr_g += image.total();
            ptr_b += 2 * image.total();
        }
        else if (image.channels() > 1) {
            ptr_g += image.total();
            ptr_r += 2 * image.total();
        }
        unsigned off = 0;
        for (int i = 0; i < image.rows; ++i) {
            Tfrom const *line = image.ptr<Tfrom const>(i);
            for (int j = 0; j < image.cols; ++j) {
                if (image.channels() > 1) {
                    ptr_b[off] = *line++;
                    ptr_g[off] = *line++;
                }
                ptr_r[off] = *line++;
                ++off;
            }
        }
        CHECK(off == image.total());
        if (image.channels() == 1) return buffer + image.total();
        return buffer + 3 * image.total();
    }

public:
    struct Config: public ImageStream::Config {
        unsigned onehot;
        unsigned batch;
        bool pad;
        Config (): onehot(0), batch(1), pad(false) {
        }
    };
    BatchImageStream (std::string const &path, Config const &c)
        : ImageStream(fs::path(path), c), onehot(c.onehot), batch(c.batch), pad(c.pad) {
        import_array();
    }
    tuple next () {
        object images, label, anno;
        float *image_buf = NULL;
        float *label_buf = NULL;
        float *anno_buf = NULL;
        unsigned loaded = 0;
        try {
            npy_intp image_dims[] = {batch, 0, 0, 0};
            npy_intp label_dims[] = {batch};
            npy_intp onehot_dims[] = {batch, onehot};
            npy_intp anno_dims[] = {batch, 0, 0, 0};
            for (unsigned i = 0; i < batch; ++i) {
                Value value = ImageStream::next();
                if (i == 0) {
                    image_dims[1] = value.image.channels();
                    image_dims[2] = value.image.rows;
                    image_dims[3] = value.image.cols;
                    images = object(boost::python::handle<>(PyArray_SimpleNew(4, image_dims, NPY_FLOAT)));
                    CHECK(images.ptr());
                    image_buf = get_ndarray_data<float>(images);
                    if (value.annotation.data) {
                        // we are doing annotation
                        anno_dims[1] = value.annotation.channels();
                        anno_dims[2] = value.annotation.rows;
                        anno_dims[3] = value.annotation.cols;
                        anno = object(boost::python::handle<>(PyArray_SimpleNew(4, anno_dims, NPY_FLOAT)));
                        CHECK(anno.ptr());
                        anno_buf = get_ndarray_data<float>(anno);
                    }
                    else if (onehot > 0) {
                        label = object(boost::python::handle<>(PyArray_SimpleNew(2, onehot_dims, NPY_FLOAT)));
                        CHECK(label.ptr());
                        label_buf = get_ndarray_data<float>(label);
                    }
                    else {
                        label = object(boost::python::handle<>(PyArray_SimpleNew(1, label_dims, NPY_FLOAT)));
                        CHECK(label.ptr());
                        label_buf = get_ndarray_data<float>(label);
                    }
                }
                else {
                    CHECK(image_buf);
                    CHECK(value.image.channels() == image_dims[1]);
                    CHECK(value.image.rows == image_dims[2]);
                    CHECK(value.image.cols == image_dims[3]);
                    if (value.annotation.data) {
                        CHECK(anno_buf);
                        CHECK(value.annotation.channels() == anno_dims[1]);
                        CHECK(value.annotation.rows == anno_dims[2]);
                        CHECK(value.annotation.cols == anno_dims[3]);
                    }
                    else {
                        CHECK(label_buf);
                    }
                }
                CHECK(value.image.type() == CV_8U
                        || value.image.type() == CV_8UC3);
                image_buf = split<uint8_t, float>(value.image, image_buf);
                if (value.annotation.data) {
                    CHECK(value.annotation.type() == CV_8U
                            || value.annotation.type() == CV_8UC3);
                    anno_buf = split<uint8_t, float>(value.annotation, anno_buf);
                }
                else if (onehot > 0) {
                    unsigned l(value.label);
                    CHECK(l == value.label);
                    std::fill(label_buf, label_buf + onehot, 0);
                    label_buf[l] = 1;
                    label_buf += onehot;
                }
                else {
                    *label_buf = value.label;
                    ++label_buf;
                }
                ++loaded;
            }
        } catch (EoS) {
            if (!pad) throw EoS();
        }
        if (loaded == 0) throw EoS();
        CHECK(loaded <= batch);
        unsigned padding = batch - loaded;
        if (!anno.is_none()) return make_tuple(images, anno, padding);
        return make_tuple(images, label, padding);
    }
};

object create_image_stream (tuple args, dict kwargs) {
    object self = args[0];
    CHECK(len(args) > 1);
    string path = extract<string>(args[1]);
    BatchImageStream::Config config;
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
#define XFER(param) \
    config.param = extract<decltype(config.param)>(kwargs.get(#param, config.param))
    // stream parameters
    XFER(cache);
    XFER(channels);
    XFER(onehot);
    XFER(batch);
    XFER(pad);
    XFER(seed);
    XFER(loop);
    XFER(shuffle);
    XFER(reshuffle);
    XFER(stratify);
    XFER(preload);
    XFER(threads);
    // image loader parameters
    XFER(mode);
    object annotate = kwargs.get("annotate");
    if (!annotate.is_none()) {
        extract<int> get_int(annotate);
        extract<string> get_str(annotate);
        if (get_int.check()) {
            config.annotate = get_int;
        }
        else if (get_str.check()) {
            string anno = get_str;
            if (anno == "none") {
                config.annotate = ImageLoader::ANNOTATE_NONE;
            }
            else if (anno == "json") {
                config.annotate = ImageLoader::ANNOTATE_JSON;
            }
            else if (anno == "image") {
                config.annotate = ImageLoader::ANNOTATE_IMAGE;
            }
            else CHECK(0) << "annotate method not recognized: " << anno;
        }
    }
    XFER(anno_type);
    XFER(anno_copy);
    XFER(anno_thickness);

    // TODO: add perturbation
#undef XFER
    if (!kwargs.get("resize").is_none()) {
        tuple resize = extract<tuple>(kwargs.get("resize"));
        if (!resize.is_none()) {
            if (len(resize) != 2) {
                CHECK(0) << "Bad resize";
            }
            config.resize.width = extract<int>(resize[0]);
            config.resize.height = extract<int>(resize[1]);
        }
    }
    return self.attr("__init__")(path, config);
};


class Writer: public FileWriter {
    void encode (PyArrayObject *image, string *) {
        
    }
public:
    Writer (string const &path): FileWriter(fs::path(path)) {
    }
    void append (float label, string const &buf) {
        Record record(label, buf);
        FileWriter::append(record);
    }

    void append (string const &buf1, string const &buf2) {
        Record record(0, buf1, buf2);
        FileWriter::append(record);
    }
};

void (Writer::*append1) (float, string const &) = &Writer::append;
void (Writer::*append2) (string const &, string const &) = &Writer::append;

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
    class_<BatchImageStream::Config>("ImageStreamParams", init<>());
    class_<BatchImageStream, boost::noncopyable>("ImageStream", no_init)
        .def("__init__", raw_function(create_image_stream), "exposed ctor")
        .def(init<string, BatchImageStream::Config const&>()) // C++ constructor not exposed
        .def("next", &BatchImageStream::next)
        .def("size", &BatchImageStream::size)
        .def("reset", &BatchImageStream::reset)
    ;
    class_<Writer>("Writer", init<string>())
        .def("append", append1)
        .def("append", append2)
    ;
}

