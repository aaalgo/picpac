#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <fstream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/raw_function.hpp>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
//#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include "picpac.h"
#include "picpac-image.h"
#include "bachelor/bachelor.h"
using namespace boost::python;
using namespace picpac;
using bachelor::NumpyBatch;
namespace {
    using std::unique_ptr;

    class FacetData {
    public:
        virtual ~FacetData () {}
        virtual void fill_next (cv::Mat const &image) = 0;
        virtual PyObject *detach () = 0;
        virtual void dump (string const &prefix) {
        }
    };

    class BachelorFacetData: public FacetData, public NumpyBatch {
    public:
        BachelorFacetData (NumpyBatch::Config const &conf): NumpyBatch(conf) {
        }

        virtual void fill_next (cv::Mat const &image) {
            NumpyBatch::fill_next(image);
        }
        virtual PyObject *detach () {
            return NumpyBatch::detach();
        }
        virtual void dump (string const &prefix) {
            CHECK(conf.order == bachelor::NHWC);
            char *p = reinterpret_cast<char *>(buffer);
            if (conf.channels ==1 || conf.channels == 3) {
                for (int i = 0; i < cnt; ++i) {
                    cv::Mat image(conf.height, conf.width, CV_MAKETYPE(conf.depth, conf.channels), p);
                    cv::imwrite(prefix + "_" + lexical_cast<string>(i) + ".png", image);
                    p += image_size;
                }
            }
        }
    };

    class FacetFeatureData: public FacetData {
        vector<cv::Mat> features;
    public:
        virtual void fill_next (cv::Mat const &image) {
            features.emplace_back(image.clone());
        }
        virtual PyObject *detach () {
            int total = 0;
            int cols = 0;
            for (auto const &v: features) {
                if (v.rows) {
                    total += v.rows;
                    if (cols) CHECK(cols == v.cols) << "feature dimension do not match";
                    else cols = v.cols;
                }
            }
            cols += 1;
            npy_intp dims[2] = {total, cols};
            PyObject *pydata = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
            CHECK(pydata);
            // copy
            float *p = (float *)PyArray_DATA(pydata);
            for (unsigned i = 0; i < features.size(); ++i) {
                cv::Mat const &v = features[i];
                if (v.rows == 0) continue;
                for (int j = 0; j < v.rows; ++j, p += cols) {
                    p[0] = i;
                    float const *from = v.ptr<float const>(j);
                    std::copy(from, from + v.cols, p+1);
                }
            }
            return pydata;
        }
    };

#if PY_MAJOR_VERSION >= 3
    object string_to_python (string &s) {
        if (s.size()) {
            return object(handle<>(PyBytes_FromStringAndSize(&s[0], s.size())));
        }
        else {
            return object(handle<>(PyBytes_FromStringAndSize(NULL, 0)));
        }
    }
#else
#define string_to_python(s) s
#endif

class PyImageStream: public ImageStream {
    int batch;
    int dump;
    int dump_cnt;
    string order;
    string colorspace;
    bachelor::Order bachelor_order;
    bachelor::ColorSpace bachelor_colorspace;
    object ctor;
public:
    struct Config: public ImageStream::Config {

        Config (dict const &kwargs) {
            boost::python::object simplejson = boost::python::import("simplejson");

            //dict sampler = kwargs.get("sampler");
#define UPDATE_CONFIG(V, D) \
            V = extract<decltype(V)>(D.get(#V, V)) 
            //UPDATE_CONFIG(seed, sampler);
            UPDATE_CONFIG(loop, kwargs);
            UPDATE_CONFIG(shuffle, kwargs);
            UPDATE_CONFIG(reshuffle, kwargs);
            UPDATE_CONFIG(stratify, kwargs);
#if 0
            split, split_fold, split_negate
            mixin, mixin_group_reset, mixin_group_delta, mixin_max
#endif


            UPDATE_CONFIG(mixin, kwargs);
            UPDATE_CONFIG(mixin_group_reset, kwargs);
            UPDATE_CONFIG(mixin_group_delta, kwargs);
            UPDATE_CONFIG(mixin_max, kwargs);

            //dict loader = kwargs.get("loader");
            UPDATE_CONFIG(cache, kwargs);
            UPDATE_CONFIG(preload, kwargs);
            UPDATE_CONFIG(threads, kwargs);
            UPDATE_CONFIG(channels, kwargs);

            if (kwargs.has_key("images")) {
                list rf = extract<list>(kwargs.get("images"));
                images.clear();
                for (int i = 0; i < len(rf); ++i) {
                    images.push_back(extract<int>(rf[i]));
                }
            }

            if (kwargs.has_key("raw")) {
                list rf = extract<list>(kwargs.get("raw"));
                for (int i = 0; i < len(rf); ++i) {
                    raw.push_back(extract<int>(rf[i]));
                }
            }

            if (kwargs.has_key("annotate")) {
                // handle annotation
                auto anno = kwargs.get("annotate");
                extract<list> e(anno);
                if (e.check()) {
                    list fields = e();
                    for (int i = 0; i < len(fields); ++i) {
                        annotate.push_back(extract<int>(fields[i]));
                    }
                }
                else {
                    extract<bool> e(anno);
                    CHECK(e.check());
                    LOG(WARNING) << "setting annotate to True is obsolete.  Use [1] or a list of annotate fields.";
                    LOG(WARNING) << "pushing 1 to annotate field list.";
                    if (e()) {
                        annotate.push_back(1);
                    }
                }
                // anno 
            }

            // check dtype
            string dt = extract<string>(kwargs.get("dtype", "uint8"));
            dtype = dtype_np2cv(dt);
            object trans = kwargs.get("transforms", list());
            transforms = extract<string>(simplejson.attr("dumps")(trans));
        }
    };

    PyImageStream (dict kwargs) //{std::string const &path, Config const &c)
        : ImageStream(fs::path(extract<string>(kwargs.get("db"))), Config(kwargs)), batch(1), dump(0), dump_cnt(0), order("NHWC"), colorspace("BGR") {
            UPDATE_CONFIG(batch, kwargs);
            UPDATE_CONFIG(dump, kwargs);
            UPDATE_CONFIG(order, kwargs);
            UPDATE_CONFIG(colorspace, kwargs);
#undef UPDATE_CONFIG
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


            auto collections = import("collections");
            auto namedtuple = collections.attr("namedtuple");
            list fields;
            fields.append("ids");       // np array
            fields.append("labels");    // labels
            fields.append("raw");    // list of field.  each field is a list of batch size, containing the field data
            ctor = namedtuple("Meta", fields);

            if (dump) {
                fs::create_directory("picpac_dump");
                LOG(WARNING) << "dumping image to picpac_dump/{batch}_{field}_{image}.png";
            }
    }

    list next () {
        // return a batch
        // if EOS, this will 
        vector<int32_t> ids;
        vector<float> labels;
        vector<list> raw_fields;
        for (auto const &x: ImageLoader::config.raw) {
            raw_fields.push_back(list());
        }
        vector<unique_ptr<FacetData>> data;
        // create batch, emplace first object
        Value v(ImageStream::next());
        ids.push_back(v.id);
        labels.push_back(v.label);
        CHECK(v.raw.size() == raw_fields.size());
        for (unsigned i = 0; i < v.raw.size(); ++i) {
            raw_fields[i].append(string_to_python(v.raw[i]));
        }
        for (auto &im: v.facets) {
            im.check_pythonize();
            if ((im.type == Facet::IMAGE) || (im.type == Facet::LABEL)) {
                //if (im.image.data) {
                NumpyBatch::Config conf;
                conf.batch = batch;
                conf.height = im.image.rows;
                conf.width = im.image.cols;
                conf.channels = im.image.channels();
                conf.depth = im.image.depth();
                conf.order = bachelor_order;
                if (im.type == Facet::IMAGE) {
                    conf.colorspace = bachelor_colorspace;
                }
                data.emplace_back(new BachelorFacetData(conf));
                data.back()->fill_next(im.image);
            }
            else if (im.type == Facet::FEATURE) {
                data.emplace_back(new FacetFeatureData());
                data.back()->fill_next(im.image);
            }
            else if (im.type == Facet::NONE) {
                data.emplace_back(nullptr);
            }
            else {
                CHECK(0);
            }
        }
        for (int i = 1; i < batch; ++i) {
            // reset of the batch
            try {
                Value v(ImageStream::next());
                ids.push_back(v.id);
                labels.push_back(v.label);
                CHECK(v.raw.size() == raw_fields.size());
                for (unsigned i = 0; i < v.raw.size(); ++i) {
                    raw_fields[i].append(string_to_python(v.raw[i]));
                }
                CHECK(data.size() == v.facets.size());
                for (unsigned j = 0; j < data.size(); ++j) {
                    auto &im = v.facets[j];
                    im.check_pythonize();
                    if (im.type == Facet::NONE) continue;
                    CHECK(data[j]);
                    data[j]->fill_next(im.image);
                }
            }
            catch (EoS const &) {
                break;
            }
        }
        npy_intp dims[1] = {labels.size()};
        PyObject *pyids = PyArray_SimpleNew(1, dims, NPY_INT32);
        CHECK(pyids);
        PyObject *pylabels = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
        CHECK(pylabels);
        std::copy(ids.begin(), ids.end(), (int32_t *)PyArray_DATA(pyids));
        std::copy(labels.begin(), labels.end(), (float *)PyArray_DATA(pylabels));

        list raw_field_list;
        for (auto &l: raw_fields) {
            raw_field_list.append(l);
        }
        list r;
        r.append(ctor(boost::python::handle<>(pyids), boost::python::handle<>(pylabels), raw_field_list));

        if (dump > 0 && dump_cnt < dump) {
            string prefix = "picpac_dump/" + lexical_cast<string>(dump_cnt);
            int fc = 0;
            for (auto &p: data) {
                if (p) {
                    p->dump(prefix + "_" + lexical_cast<string>(fc));
                }
                fc += 1;
            }
            dump_cnt += 1;
        }
        for (auto &p: data) {
            if (p) {
                r.append(object(boost::python::handle<>(p->detach())));
            }
            else {
                r.append(object(boost::python::handle<>(Py_None)));
            }
        }
        return r;
    }
};

object return_iterator (tuple args, dict kwargs) {
    object self = args[0];
    self.attr("reset")();
    return self;
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

    const_buffer pyobject2buffer (PyObject *buf) {
#if PY_MAJOR_VERSION >= 3
        if (PyBytes_Check(buf)) {
            return const_buffer(PyBytes_AsString(buf), PyBytes_Size(buf));
        }
#else
        if (PyString_Check(buf)) {
            return const_buffer(PyString_AsString(buf), PyString_Size(buf));
        }
#endif
        CHECK(0) << "can only append string or bytes";

    }

    void append (float label, PyObject *buf) {
        Record record(label, pyobject2buffer(buf));
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }

    void append (float label, PyObject *buf, PyObject *buf2) {
        Record record(label, pyobject2buffer(buf), pyobject2buffer(buf2));
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }

    void append (float label, PyObject *buf, PyObject *buf2, PyObject *buf3) {
        Record record(label, pyobject2buffer(buf), pyobject2buffer(buf2), pyobject2buffer(buf3));
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }

    void append (float label, PyObject *buf, PyObject *buf2, PyObject *buf3, PyObject *buf4) {
        Record record(label, pyobject2buffer(buf), pyobject2buffer(buf2), pyobject2buffer(buf3), pyobject2buffer(buf4));
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }

#if 0
    void append (float label, PyObject *buf, PyObject *buf2, PyObject *buf3, PyObject *buf4) {
        Record record(label, pyobject2buffer(buf), pyobject2buffer(buf2), pyobject2buffer(buf3), pyobject2buffer(buf4));
        record.meta().id = nextid;
        ++nextid;
        FileWriter::append(record);
    }
#endif

#if 0
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
#endif
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
#if PY_MAJOR_VERSION >= 3
            const_buffer buf = rec.field(i);
            fields.append(object(handle<>(PyBytes_FromStringAndSize(boost::asio::buffer_cast<char const *>(buf), boost::asio::buffer_size(buf)))));
#else
            fields.append(rec.field_string(i));
#endif
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

#if PY_MAJOR_VERSION >= 3
object
#else
string
#endif
encode_raw_ndarray (object &obj) {
    std::ostringstream ss;
    serialize_raw_ndarray(obj, ss);
#if PY_MAJOR_VERSION >= 3
    string str = ss.str();
    return object(handle<>(PyBytes_FromStringAndSize(&str[0], str.size())));
#else
    return ss.str();
#endif
}

void write_raw_ndarray (string const &path, object &obj) {
    std::ofstream os(path.c_str(), std::ios::binary);
    serialize_raw_ndarray(obj, os);
}

void (Writer::*append1) (float, PyObject *) = &Writer::append;
void (Writer::*append2) (float, PyObject *, PyObject *) = &Writer::append;
void (Writer::*append3) (float, PyObject *, PyObject *, PyObject *) = &Writer::append;
void (Writer::*append4) (float, PyObject *, PyObject *, PyObject *, PyObject *) = &Writer::append;
/*

void (Writer::*append4) (float, string const &, string const &, string const &) = &Writer::append;

void (Writer::*append5) (float, string const &, string const &, string const &, string const &) = &Writer::append;

void (Writer::*append6) (float, string const &, string const &, string const &, string const &, string const &) = &Writer::append;
*/

void translate_eos (EoS const &)
{
    // Use the Python 'C' API to set up an exception object
    PyErr_SetNone(PyExc_StopIteration);
}



}

namespace {
	cv::Mat Py3DArray2CvMat (PyObject *array_) {
        PyArrayObject *array((PyArrayObject *)array_);
        if (array->nd != 2 and array->nd != 3) throw runtime_error("not 3d array");
        if (array->descr->type_num != NPY_FLOAT32) throw runtime_error("not float32 array");
        if (!PyArray_ISCONTIGUOUS(array)) throw runtime_error("not contiguous");
        int dims = 1;
        if (array->nd == 3) {
            dims = array->dimensions[2];
        }
		return cv::Mat(array->dimensions[0],
					   array->dimensions[1],
					   CV_32FC(dims), array->data);
						
	}

    struct Circle {

        static unsigned constexpr PARAMS= 3;

        struct Shape {
            float x, y, r;
        };

        static void update_shape (Shape *s, cv::Point_<float> const &pt, float const *params) {
            s->x = pt.x + params[0];
            s->y = pt.y + params[1];
            s->r = params[2];

        }
        static void update_params (Shape const &c, float *params) {
            params[0] = c.x;
            params[1] = c.y;
            params[2] = c.r;
        }

        static float overlap (Shape const &a, Shape const &b) {
            float dx = a.x - b.x;
            float dy = a.y - b.y;
            float d = sqrt(dx * dx + dy * dy);
            float r = std::max(a.r, b.r) + 1;
            return  (r-d)/r;
        }

        static void draw (cv::Mat image, Shape const &c) {
            int r = std::round(c.r);
            int x = std::round(c.x);
            int y = std::round(c.y);
            cv::circle(image, cv::Point(x, y), r, cv::Scalar(255, 0, 0), 1);
        }
    };

    struct Box {
        static unsigned constexpr PARAMS= 4;

        typedef cv::Rect_<float> Shape;

        static void update_shape (Shape *s, cv::Point_<float> const &pt, float const *params) {
            s->x = pt.x + params[0] - params[2]/2;
            s->y = pt.y + params[1] - params[3]/2;
            s->width = params[2];
            s->height = params[3];
        }

        static float overlap (Shape const &s1, Shape const &s2) {
            float o = (s1 & s2).area();
            return o / (s1.area() + s2.area() - o +1);
        }

        static void update_params (Shape const &s, float *params) {
            params[0] = s.x;
            params[1] = s.y;
            params[2] = s.x + s.width;
            params[3] = s.y + s.height;
        }

        static void draw (cv::Mat image, Shape const &c) {
            cv::rectangle(image, cv::Point(int(round(c.x)), int(round(c.y))),
                                 cv::Point(int(round(c.x+c.width)), int(round(c.y+c.height))), 
                                 cv::Scalar(255, 0, 0), 1);
        }
    };

    template <typename SHAPE>
    class AnchorProposal {
        int upsize;
        float pth;
        float th;

        struct Shape: public SHAPE::Shape {
            float score;
            float keep;
        };
    public:
        AnchorProposal (int up, float pth_, float th_): upsize(up), pth(pth_), th(th_) {
        }

        PyObject* apply (PyObject *prob_, PyObject *params_, PyObject *image_) {
            cv::Mat prob(Py3DArray2CvMat(prob_));
            cv::Mat params(Py3DArray2CvMat(params_));

            CHECK(prob.type() == CV_32FC1);
            //CHECK(params.type() == CV_32FC3);
            CHECK(prob.rows == params.rows);
            CHECK(prob.cols == params.cols);
            //CHECK(prob.channels() == 1);
            CHECK(params.channels() == SHAPE::PARAMS * prob.channels());
            vector<Shape> all;
            int priors = prob.channels();
            for (int y = 0; y < prob.rows; ++y) {
                float const *pl = prob.ptr<float const>(y);
                float const *pp = params.ptr<float const>(y);
                for (int x = 0; x < prob.cols; ++x) {
                    cv::Point_<float> pt(x * upsize, y * upsize);
                    for (int prior = 0; prior < priors; ++prior, ++pl, pp += SHAPE::PARAMS) {
                        if (pl[0] < pth) continue;
                        Shape c;
                        SHAPE::update_shape(&c, pt, pp);
                        c.score = pl[0];
                        c.keep = true;
                        all.push_back(c);
                    }
                }
            }
            sort(all.begin(), all.end(), [](Shape const &a, Shape const &b){return a.score > b.score;});

            unsigned cnt = 0;
            for (unsigned i = 0; i < all.size(); ++i) {
                if (!all[i].keep) continue;
                cnt += 1;
                Shape const &a = all[i];
                for (unsigned j = i+1; j < all.size(); ++j) {
                    Shape &b = all[j];
                    float d = SHAPE::overlap(a, b);
                    if (d > th) {
                       b.keep = false;
                    }
                }
            }
		
        	npy_intp dims[2] = {cnt, SHAPE::PARAMS};
            PyObject *result = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
            float *out_params = (float *)((PyArrayObject *)result)->data;
            for (auto const &c: all) {
                if (!c.keep) continue;
                SHAPE::update_params(c, out_params);
                out_params += SHAPE::PARAMS;
            }

            if (image_ != Py_None) {
                cv::Mat image(Py3DArray2CvMat(image_));
                for (auto const &c: all) {
                    if (!c.keep) continue;
                    SHAPE::draw(image, c);
                }
            }
            return result;
        }
    };
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
    scope().attr("OVERWRITE") = Writer::FLAG_OVERWRITE;
#ifdef CVBOOSTCONVERTER_HPP_
    to_python_converter<cv::Mat,
                     pbcvt::matToNDArrayBoostConverter>();

    pbcvt::matFromNDArrayBoostConverter();
#endif
    register_exception_translator<EoS>(&translate_eos);
    class_<PyImageStream, boost::noncopyable>("ImageStream", init<dict>())
        .def("__iter__", raw_function(return_iterator))
#if (PY_VERSION_HEX >= 0x03000000)
        .def("__next__", &PyImageStream::next)
#endif
        .def("next", &PyImageStream::next)
        .def("size", &PyImageStream::size)
        .def("reset", &PyImageStream::reset)
        //.def("categories", &PyImageStream::categories)
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
    class_<Writer>("Writer", init<string, int>())
        .def("append", append1)
        .def("append", append2)
        .def("append", append3)
        .def("append", append4)
        /*
        .def("append", append5)
        .def("append", append6)
        */
        .def("setNextId", &Writer::setNextId);
    ;
    def("encode_raw", ::encode_raw_ndarray);
    def("write_raw", ::write_raw_ndarray);

    class_<AnchorProposal<Circle>>("CircleProposal", init<int, float, float>())
        .def("apply", &AnchorProposal<Circle>::apply)
    ;
    class_<AnchorProposal<Box>>("BoxProposal", init<int, float, float>())
        .def("apply", &AnchorProposal<Box>::apply)
    ;
//#undef NUMPY_IMPORT_ARRAY_RETVAL
//#define NUMPY_IMPORT_ARRAY_RETVAL
}

