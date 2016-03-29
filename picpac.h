#ifndef PICPAC_INCLUDE
#define PICPAC_INCLUDE
#include <array>
#include <queue>
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <random>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <glog/logging.h>

namespace picpac {

    using std::array;
    using std::queue;
    using std::vector;
    using std::string;
    using std::numeric_limits;
    using std::runtime_error;
    using boost::lexical_cast;

    namespace fs = boost::filesystem;

    // maximal number of fields per record
    static constexpr unsigned MAX_FIELDS = 6;
    // maximal number of records per segment
    static constexpr unsigned MAX_SEG_RECORDS = 1020;
    // record alignment for faster access
    static constexpr unsigned RECORD_ALIGN = 4096;
    static constexpr size_t MAX_RECORD_SIZE = 512*1024*1024;  // 512MB

    static constexpr int MAX_CATEGORIES = 2000;
    static constexpr unsigned DEFAULT_BUFFER = 128;

    enum {  // Record field type
        FIELD_NONE = 0,
        /*
        FIELD_FILE = 1,
        FIELD_TEXT = 2,
        FIELD_OTHER = 3 
        */
    };

    class BadLabel: public runtime_error {
    public:
        BadLabel (int l): runtime_error(lexical_cast<string>(l)) {}
    };

    class BadFile: public runtime_error {
    public:
        BadFile (fs::path const &p): runtime_error(p.native()) {}
    };

    class BadRecordSize: public runtime_error {
    public:
        BadRecordSize (uintmax_t sz): runtime_error(lexical_cast<string>(sz)) {}
    };

    /// Essential meta data of an image
    struct __attribute__((__packed__)) Meta { 
        struct __attribute__((__packed__)) Field {  // 8 bytes
            uint32_t size;
            uint8_t type;
            uint8_t reserved1;
            uint16_t reserved2;
        };
        // total 16 bytes
        uint32_t id;     // user provided ID
        float label;
        uint8_t width;    // # fields
        uint8_t reserved1;
        uint16_t reserved2;
        uint32_t reserved3;
        std::array<Field, MAX_FIELDS> fields;
        void init () {
            char *begin = reinterpret_cast<char *>(this);
            std::fill(begin, begin + sizeof(*this), 0);
        }
    };
    static_assert(sizeof(Meta) == 64, "bad Meta size");

    /// Image Record
    /**
     */
    struct Record {     // record owns the data
        string data;    // raw data
        Meta *meta;     // pointer into data
        array<char *, MAX_FIELDS> fields;   // pointers into data
        ssize_t write (int fd) const;
        ssize_t read (int fd, off_t off, size_t size);
        void pack (double label, fs::path const &image);
        void pack (double label, fs::path const &image, string const &extra);
    private:
        void alloc_helper (int nf, uintmax_t off) {
            if (!(off < MAX_RECORD_SIZE)) {
                throw BadRecordSize(off);
            }
            data.resize(off);
            meta = reinterpret_cast<Meta *>(&data[0]);
            meta->init();
            meta->width = nf;
        }

        template <typename ... Args>
        void alloc_helper (int ifld, uintmax_t off, uintmax_t size, Args... args) {
            alloc_helper(ifld + 1, off + size, args...);
            meta->fields[ifld].size = size;
            fields[ifld] = &data[off];
        }

        template <typename ... Args>
        void alloc (double label, Args... args) {
            alloc_helper(0, sizeof(Meta), args...);
            meta->label = label;
        }
    };

    struct __attribute__((__packed__)) SegmentHeader {
        static uint16_t constexpr MAGIC = 0x59AC;
        static uint16_t constexpr VERSION = 0x0100; // major 01, minor 00
        uint16_t magic;     
        uint16_t version;
        uint16_t size;      // number of records
        uint16_t reserved1;
        uint64_t link;      // next segment offset
        // -- 16 bytes so far
        uint64_t reserved2;
        uint64_t reserved3;
        // -- 8160 bytes below
        array<uint32_t, MAX_SEG_RECORDS> sizes;
        array<float, MAX_SEG_RECORDS> labels;

        void init () {
            char *begin = reinterpret_cast<char *>(this);
            std::fill(begin, begin + sizeof(*this), 0);
            magic = MAGIC;
            version = VERSION;
        }
    };

    static_assert(sizeof(SegmentHeader) == 8192, "Bad segment header size");
    static_assert(sizeof(SegmentHeader) % RECORD_ALIGN == 0, "Bad segment header size");

    class FileWriter {
        int fd;
        off_t seg_off;      // last segment offset
        SegmentHeader seg;  // segment header
        unsigned next;      // next record offset within segment
        // initialize a new segment at the end of file
        // and set status to the new segment
        void open_segment ();
        // write the meta data of last segment to file
        void close_segment ();
    public:
        FileWriter (fs::path const &path);
        ~FileWriter ();
        void append (Record const &r);
    };

    struct Locator {
        off_t offset;
        uint32_t size;
        float label;
    };

    class FileReader {
        int fd;
    public:
        FileReader (fs::path const &path);
        ~FileReader ();
        void ping (vector<Locator> *l);
        void read (Locator const &l, Record *r) {
            ssize_t sz = r->read(fd, l.offset, l.size);
            CHECK(sz == l.size);
        }
    };

    class IndexedFileReader: public FileReader {
        vector<Locator> index;
    public:
        IndexedFileReader (fs::path const &path)
            : FileReader(path) {
            ping(&index);
        }
        size_t size () const { return index.size(); }
        void read (size_t i, Record *r) {
            if (!(i < index.size())) throw std::out_of_range("");
            FileReader::read(index[i], r);
        }
    };

    struct EoS {
    };

    class BasicStreamer: public FileReader {
    public:
        struct Config {
            int seed;           // random seed
            bool loop;   
            bool shuffle;
            bool reshuffle;
            bool stratify;
            unsigned buffer;
            unsigned threads;   // 0 to use all cores

            unsigned splits;
            vector<unsigned> keys;   // split keys to include

            Config()
                : seed(2016),
                loop(true),
                shuffle(true),
                reshuffle(true),
                stratify(true),
                buffer(DEFAULT_BUFFER),
                threads(0),
                splits(1),
                keys{0} {
            }
                    
        };

        struct KFoldConfig: public Config {
            KFoldConfig (unsigned K, unsigned fold, bool train);
        };
    protected:
        Config config;
        std::default_random_engine rng;
    private:
        struct Group {
            unsigned id;    // unique group ID
            vector<Locator> index;
            unsigned next;
        };
        vector<Group> groups;
        unsigned next_group;
    public:
        BasicStreamer (fs::path const &, Config const &);
        Locator next ();
        void read_next (Record *r) {
            read(next(), r);
        }
    };

    template <typename TR> // transform class to serve as base class
    class TransformStreamer: public BasicStreamer, public TR {
        typedef BasicStreamer::Config StreamerConfig;
        typedef typename TR::Config TransformConfig;
        typedef typename TR::Value Value;
        typedef typename TR::Perturb Perturb;
        typedef std::lock_guard<std::mutex> lock_guard;
        typedef std::unique_lock<std::mutex> unique_lock;
        struct Task {
            Locator locator;
            Perturb perturb;
        };
        queue<Task> todo;
        queue<Value> done;
        int inqueue; // todo.size() + done.size()
        bool eos;    // eos signal from upstream
        std::condition_variable has_todo;
        std::condition_variable has_done;
        std::mutex mutex;
        vector<std::thread> threads;
        static unsigned detect_threads () {
            return 4;
        }

        void worker () {
            for (;;) {
                Task task;
                {
                    unique_lock lock(mutex);
                    while (todo.empty()) {
                        if (eos) return;
                        has_todo.wait(lock);
                    }
                    task = todo.front();
                    todo.pop();
                }
                Record r;
                FileReader::read(task.locator, &r);
                Value v(TR::transform(r, task.perturb));
                if (!(v >= 0 && v < 17)) {
                    std::cerr << task.locator.offset << '\t' << task.locator.size << '\t' << task.locator.label << std::endl;
                    *(char *)0 = 0;;
                }
                {
                    lock_guard lock(mutex);
                    done.push(std::move(v));
                    has_done.notify_one();
                }
            }
        }

        bool prefetch_unsafe () {
            try {
                Task task;
                task.locator = BasicStreamer::next();
                task.perturb = TR::perturb(rng);
                todo.push(task);
                ++inqueue;
                has_todo.notify_one();
                return true;
            }
            catch (EoS) {
                eos = true;
                has_todo.notify_all();
                return false;
            }
        }
    public:
        TransformStreamer (fs::path const &p, StreamerConfig const &c, TransformConfig const &c2)
            : BasicStreamer(p, c), TR(c2), inqueue(0), eos(false) {
            // enqueue tasks
            for (unsigned i = 0; i < c.buffer; ++i) {
                if (!prefetch_unsafe()) break;
            }
            unsigned nth = c.threads;
            if (nth == 0) {
                nth = detect_threads();
            }
            LOG(INFO) << "Starting " << nth << " threads.";
            for (unsigned i = 0; i < nth; ++i) {
                threads.emplace_back([this](){this->worker();});
            }
        }

        ~TransformStreamer () {
            eos = true;
            has_todo.notify_all();
            for (auto &th: threads) {
                th.join();
            }
        }

        Value next () {
            unique_lock lock(mutex);
            prefetch_unsafe();
            while (done.empty()) {
                if (inqueue == 0) {
                    throw EoS();
                }
                has_done.wait(lock);
            }
            Value v(std::move(done.front()));
            done.pop();
            --inqueue;
            return v;
        }
    };
}

#endif
