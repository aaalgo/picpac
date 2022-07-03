#pragma once
#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <random>
#include <functional>
#include <filesystem>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#define SPDLOG_FMT_EXTERNAL
#include <spdlog/spdlog.h>
#define CHECK(x) do { if (!(x)) { logging::critical("{}:{} {} assertion failed: {}", __FILE__, __LINE__, __FUNCTION__,  #x); abort(); } } while(0)


namespace picpac {


    using std::array;
    using std::vector;
    using std::string;
    using std::string_view;
    using std::ifstream;
    using std::numeric_limits;
    using std::runtime_error;
    typedef std::lock_guard<std::mutex> lock_guard;
    typedef std::unique_lock<std::mutex> unique_lock;
    typedef std::default_random_engine random_engine;

    namespace fs = std::filesystem;
    namespace logging = spdlog;


    /// Static coded maximal number of fields per record
    static constexpr unsigned MAX_FIELDS = 6;   // this can be increased
    /// Static coded segment header size
    static constexpr unsigned SEGMENT_HEADER_SIZE = 8192;
    /// Static coded maximal number of records per segment
    /**
     * This must be carefully calculated so that the segment
     * header struct adds up to SEGMENT_HEADER_SIZE.
     */
    static constexpr unsigned MAX_SEG_RECORDS = 1020;
    /// Record alignment for faster access
    static constexpr unsigned RECORD_ALIGN = 4096;
    /// Static coded maximal record size
    static constexpr size_t MAX_RECORD_SIZE = 512*1024*1024;  // 512MB
    static_assert(MAX_RECORD_SIZE < numeric_limits<int32_t>::max(), "record too large");
    /// Maximal number of categories.
    static constexpr unsigned MAX_CATEGORIES = 2000;
    /* Maximal category ID is (MAX_CATEGORIES - 1)
     * we need to make sure this can be stored in float without
     * loss of precision.  If it's too big, the LSB 1 will be lost.
     */
    static constexpr unsigned MAX_CATEGORY_TEST = (MAX_CATEGORIES - 1) | 1;
    static_assert(float(MAX_CATEGORY_TEST) == MAX_CATEGORY_TEST, "too many categories");
    static constexpr int DEFAULT_SEED = 2016;
    static constexpr unsigned DEFAULT_PRELOAD = 256;
    static constexpr unsigned DEFAULT_THREADS = 4;

    // format stack trace
    class Stack: public std::vector<char const *> {
        static const unsigned MAX_BACKTRACE = 100;
        char **symbols;
    public:
        Stack ();
        ~Stack ();
        string format (std::string const &prefix = "") const;
    };

    enum FieldType {  // Record field type
        FIELD_NONE = 0,
        FIELD_IMAGE = 1,
        FIELD_ANNOTATION_IMAGE = 2,
        FIELD_ANNOTATION_JSON = 3,
        CHECK_FIELD_SIZE
    };
    static_assert(CHECK_FIELD_SIZE - 1 <= numeric_limits<uint8_t>::max(), "Too many field types");

    class BadLabel: public runtime_error {
    public:
        BadLabel (int l): runtime_error(fmt::format("Bad label: {}", l)) {
        }
    };

    class BadFile: public runtime_error {
    public:
        BadFile (fs::path const &p): runtime_error(fmt::format("Bad path: {}", p.native())) {}
    };

    class DataCorruption: public runtime_error {
    public:
        DataCorruption (): runtime_error("picpac data corruption") {}
    };

    class BadRecordSize: public runtime_error {
    public:
        BadRecordSize (uintmax_t sz): runtime_error(fmt::format("Bad record size: {}", sz)) {}
    };

    /// Meta data of a record
    struct __attribute__((__packed__)) Meta { 
        struct __attribute__((__packed__)) Field {  // 8 bytes
            /// Field size
            uint32_t size;
            /// Field type.  See FieldType.
            uint8_t type;
            uint8_t reserved1;
            uint16_t reserved2;
        };
        // total 16 bytes
        /// For storing user ID. PicPac does not use the ID field.
        uint32_t id;
        /// Label of record -- if it is relevant.
        /** Label can be an integer representing category ID,
         * or a float number for regression.  In the previous case,
         * number of category must not exceed system limitation.
         */
        float label;
        /// Number of fields in the record.
        uint8_t width; 
        uint8_t reserved1;
        int16_t label2;  // optional secondary label, for stratification
        uint32_t reserved2; 
        /// Meta data of fields.
        std::array<Field, MAX_FIELDS> fields;

        void init () {
            char *begin = reinterpret_cast<char *>(this);
            std::fill(begin, begin + sizeof(*this), 0);
        }

        // only copies user info
        void copy (Meta const &m) {
            id = m.id;
            label = m.label;
            reserved1 = m.reserved1;
            label2 = m.label2;
            reserved2 = m.reserved2;
        }
    };
    static_assert(sizeof(Meta) == 64, "bad Meta size");

    extern char const *EMPTY_BUFFER;

    /// Data Record, non-copiable but movable
    class Record {     // record owns the data
        // All field data are stored in raw/on-disk format in the data field
        string data;    // raw data
        // and meta_ptr and field_ptrs are used to access the data
        Meta *meta_ptr;     // pointer into data
        array<char *, MAX_FIELDS> field_ptrs;   // pointers into data

        void alloc_helper (int nf, uintmax_t off) {
            if (!(off < MAX_RECORD_SIZE)) {
                throw BadRecordSize(off);
            }
            data.resize(off);
            meta_ptr = reinterpret_cast<Meta *>(&data[0]);
            meta_ptr->init();
            meta_ptr->width = nf;
        }

        template <typename ... Args>
        void alloc_helper (int ifld, uintmax_t off, uintmax_t size, Args... args) {
            alloc_helper(ifld + 1, off + size, args...);
            meta_ptr->fields[ifld].size = size;
            field_ptrs[ifld] = &data[off];
        }

        template <typename ... Args>
        void alloc (double label, Args... args) {
            alloc_helper(0, sizeof(Meta), args...);
            meta_ptr->label = label;
        }

    public:
        Record (const Record&) = delete;
        Record& operator=(const Record&) = delete;

        void swap (Record &r) {
            data.swap(r.data);
            std::swap(meta_ptr, r.meta_ptr);
            std::swap(field_ptrs, r.field_ptrs);
        }

        Record (Record &&r) {
            swap(r);
        }

        Record& operator=(Record &&r) {
            swap(r);
            return *this;
        }

        ssize_t write (int fd, bool compact) const;
        ssize_t read (int fd, off_t off, size_t size);
        /// Construct an empty record, for future read from disk.
        Record () {}
        Record (float label, string_view);
        /// Construct a record with file content.
        Record (float label, fs::path const &file);

        Record (float label, string_view, string_view);
        Record (float label, string_view, string_view, string_view);
        Record (float label, string_view, string_view, string_view, string_view);
        /// Construct a record with file content and extra string.
        Record (float label, fs::path const &file, string const &extra);
        /// Construct a record with file content and extra string.
        Record (float label, fs::path const &file, fs::path const &file2);
        /// Construct a record with file content and extra string.
        Record (float label, string const &data);
        /// Construct a record with file content and extra string.
        Record (float label, string const &data, string const &extra);
        /// Construct a record with file content and extra string.
        Record (float label, string const &data, string const &extra, string const &extra2);
        Record (float label, string const &data, string const &extra, string const &extra2, string const &extra3);
        Record (float label, string const &data, string const &extra, string const &extra2, string const &extra3, string const &extra4);

        Meta &meta () { return *meta_ptr; }
        Meta const &meta () const { return *meta_ptr; }
        /// Return number fields.
        unsigned size () const { return meta_ptr->width; }

        // replace existing field to buf
        // if type >= 0, type value is also replaced
        // because a record is compactly stored, replacing a field
        // needs to reallocate the storage and copy the existing data over
        void replace (unsigned f, string const &buf, int type = -1);
        /// Get field buffer.
        string_view field (unsigned f) const {
            if (f < meta_ptr->width) {
                return string_view(field_ptrs[f], meta_ptr->fields[f].size);
            }
            return string_view(EMPTY_BUFFER, 0);
        }

        string field_string (unsigned f) const {
            if (f < meta_ptr->width) {
                return string(field_ptrs[f], field_ptrs[f] + meta_ptr->fields[f].size);
            }
            return string();
        }

        /// Get field type.
        FieldType fieldType (unsigned f) const {
            if (f < meta_ptr->width) {
                return FieldType(meta_ptr->fields[f].type);
            }
            return FIELD_NONE;
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
        array<float, MAX_SEG_RECORDS> groups;

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
        int flags;
        off_t seg_off;          // last segment offset
        SegmentHeader seg;      // segment header
        unsigned next;          // next record offset within segment
        // initialize a new segment at the end of file
        // and set status to the new segment
        void open_segment ();
        // write the meta data of last segment to file
        void close_segment ();
        bool compact () const {
            return !!(flags & COMPACT);
        }
    public:
        enum {
            INDEX_LABEL2 = 1,   // use "label" by default
                                // this makes segment header store label2
            COMPACT = 2,
            OVERWRITE = 4,
        };
        FileWriter (fs::path const &path, int flags_ = 0);
        ~FileWriter ();
        void append (Record const &r);
    };

    struct Locator {
        off_t offset;
        uint32_t size;
        float group;
        uint32_t serial;
        // A stream might have multiple underlying files
        // and these are distinguished with file
        uint32_t file;
    };

    /// A callback function that reads the record.
    typedef std::function<void(Record *)> RecordReader;

    class FileReader {
        int fd;
    public:
        FileReader (fs::path const &path);
        ~FileReader ();
        void ping (vector<Locator> *l, uint32_t file = 0);
        void read (Locator const &l, Record *r) {
            ssize_t sz = r->read(fd, l.offset, l.size);
            CHECK(sz == l.size);
        }
        /*
        RecordReader reader (Locator l) {
            return [this, l](Record *r){read(l, r);};
        }
        */
    };

    class IndexedFileReader: public FileReader {
        vector<Locator> index;
    public:
        IndexedFileReader (fs::path const &path)
            : FileReader(path) {
            ping(&index);
        }
        size_t size () const { return index.size(); }
        float group (size_t i) const {
            if (!(i < index.size())) throw std::out_of_range("");
            return index[i].group;
        }
        void read (size_t i, Record *r) {
            if (!(i < index.size())) throw std::out_of_range("");
            FileReader::read(index[i], r);
        }
        Locator const &locator (size_t i) const {
            return index[i];
        }
        void loopIndex (std::function<void(Locator const &)> cb) {
            for (auto const &l: index) {
                cb(l);
            }
        }
        void loop (std::function<void(Record &)> cb) {
            Record rec;
            for (auto const &l: index) {
                FileReader::read(l, &rec);
                cb(rec);
            }
        }
    };

    /// End of Stream exception, thrown when no more data is loaded
    struct EoS {
    };

    class Stream: public FileReader {
    public:
        struct Config {
            int seed;           // random seed
            bool loop;   
            bool shuffle;
            bool reshuffle;
            int stratify;
            bool oversample;

            unsigned split;
            vector<unsigned> split_keys; 
            int split_fold;
            bool split_negate;
            string mixin;
            float mixin_group_reset;
                                // set label to this no matter what original label is
            float mixin_group_delta;
                                // add this to label
            unsigned mixin_max; // maximal number to read from mixin
                                // no shuffling, just use the initial
                                // mixin_max items
                                //
            bool mixin_randomize = false;
                                // set to true to shuffle items
                                // before truncating mixin

            Config()
                : seed(DEFAULT_SEED),
                loop(true),
                shuffle(true),
                reshuffle(true),
                stratify(false),
                oversample(false),    // stratified sampling for cross validation
                                // but pool back into one group for streaming
                                // so smaller categories are not oversampled
                split(1),
                split_fold(0),
                split_negate(false),
                mixin_group_reset(-1),
                mixin_group_delta(0),
                mixin_max(0)
            {
            }
            /// Initialize split scheme for K-fold cross validation.
            /**
             * if train:
             *      use K-1 splits other than fold
             *  
             * if not train:
             *      use 1 split specified by fold
             */
            //void kfold ();
        };

    protected:
        Config config;
        std::default_random_engine rng;
    private:
        vector<FileReader *> readers;
        struct Group {
            unsigned id;    // unique group ID
            vector<Locator> index;
            unsigned next;
        };
        vector<Group> groups;
        vector<unsigned> group_index;
        unsigned next_group;
        unsigned sz_total;
        unsigned sz_used;
        unsigned ncat;
        unsigned ngroup;
    public:
        Stream (fs::path const &, Config const &);
        ~Stream ();
        unsigned categories () const {
            return ncat;
        }
        void reset () {
            group_index.clear();
            for (unsigned i = 0; i < groups.size(); ++i) {
                groups[i].next = 0;
                group_index.push_back(i);
            }
            next_group = 0;
        }

        Locator next ();

        RecordReader reader (Locator l) {
            return [this, l](Record *r){this->readers[l.file]->read(l, r);};
        }

        void read_next (Record *r) {
            Locator l = next();
            CHECK(l.file >= 0 && l.file < readers.size());
            readers[l.file]->read(l, r);
        }
        // return total records in the file
        unsigned total () const {
            return sz_total;
        }
        unsigned size () const {
            return sz_used;
        }
    };

    /// Dummy loader that directly returns the record itself.
    /** All loaders must follow exactly the same interface as DummyLoader:
     *  - define Config, Value and PerturbVector.
     *  - implement sample and load
     */
    class DummyLoader {
    public:
        struct Config {
        };
        typedef Record Value;
        struct CacheValue {
        };
        struct PerturbVector {
        };
        DummyLoader (Config const &) {}
        /// Sample a perturb vector.
        /** This is guaranteed to be run in serial. */
        template <typename RNG>
        void sample (RNG &, PerturbVector *) {
        }
        /// Convert a record into value.
        /** This might be invoked in parallel and should be deterministic.
         * All randomization should be done in sample. */
        void load (RecordReader rr, PerturbVector const &, Value *out,
                   CacheValue *, std::mutex *) const {
            Record r;
            rr(&r);
            *out = std::move(r);
        }
    };


    /// Stream with prefetch and transformation.
    /**
     * This stream does parallel prefetching and transformation.
     * To plugin in a transformation, parameterize this class with a
     * Loader class.  This stream preserves the order of underlying stream
     * for reproducibility.  All randomization are done in serial.
     */
    template <typename Loader = DummyLoader> // transform class to serve as base class
    class PrefetchStream: public Stream, public Loader {
    public:
        typedef typename Loader::Value Value;
        typedef typename Loader::CacheValue CacheValue;
        typedef typename Loader::PerturbVector PerturbVector;
    private:
        struct Task {       // prefetch task
            enum Status {
                EMPTY = 0,
                PENDING,
                LOADING,
                LOADED
            } status;
            // Task state transform graph:
            // empty -> pending -> loading -> loaded -> empty
            Locator locator;
            PerturbVector perturb;
            Value value;
            Task (): status(EMPTY) {
            }
        };
        bool started;
        unsigned nth;           // # threads
        bool eos;               // eos signal from upstream
        int inqueue;            // pending + loaded
        // About peek & prefetch:
        // The basic work flow is
        //      - add a prefetch task
        //      - consume a record
        // So that # tasks in queue is always the same.
        // When peeking is involved, a prefetch task might be
        // added without consuming a record.  So the follow becomes:
        // 
        // * Peak
        //      - if not previous peeked, add a prefetch task
        //      - return a record without consume it
        // * next:
        //      - if not previous peeked, add a prefetch task
        //      - remove the peek status
        //      - consume a record.
        // When multiple peeks are done consecutively, they see
        // the same record. 
        bool peeked;            // peek has been invoked
        vector<Task> queue;     // prefetch queue
        vector<CacheValue> cache;
        unsigned next_loaded;
        unsigned next_pending;
        unsigned next_empty;
        std::condition_variable has_pending;
        std::condition_variable has_loaded;
        std::mutex mutex;
        std::mutex cache_mutex;
        vector<std::thread> threads;
        Value value_holder;

        bool prefetch_unsafe () {
            if (eos) return false;
            try {
                Task &task = queue[next_empty];
                CHECK(task.status == Task::EMPTY);
                task.locator = Stream::next();
                Loader::sample(rng, &task.perturb);
                task.status = Task::PENDING;
                next_empty = (next_empty + 1) % queue.size();
                ++inqueue;
                has_pending.notify_one();
                return true;
            }
            catch (EoS) {
                eos = true;
                has_pending.notify_all();
                return false;
            }
        }

        void worker () {
            for (;;) {
                unsigned todo = 0;
                {
                    unique_lock lock(mutex);
                    while (queue[next_pending].status != Task::PENDING) {
                        if (eos) return;
                        has_pending.wait(lock);
                    }
                    todo = next_pending;
                    next_pending = (next_pending + 1) % queue.size();
                }
                Task &task = queue[todo];
                task.status = Task::LOADING;

                CacheValue *pc = nullptr;
                std::mutex *pm = nullptr;
                if (cache.size() > 0) {
                    CHECK(task.locator.serial < cache.size());
                    pc = &cache[task.locator.serial];
                    pm = &cache_mutex;
                }
                try {
                    Loader::load(reader(task.locator), task.perturb, &task.value, pc, pm);
                }
                catch (runtime_error const &e) {
                    logging::error("runtime_error: {}", e.what());
                    logging::error(Stack().format());
                    throw;
                }
                catch (...) {
                    logging::error("unknown_error");
                    logging::error(Stack().format());
                    throw;
                }
                task.status = Task::LOADED;
                // add memory barrier
                has_loaded.notify_one();
            }
        }

        void start () {
            CHECK(!started);
            started = true;
            eos = false;
            inqueue = 0;
            peeked = false;
            next_loaded = next_pending = next_empty = 0;
            CHECK(queue.size());
            for (auto &v: queue) {
                v.status = Task::EMPTY;
            }
            for (unsigned i = 0; i < queue.size() - 1; ++i) {
                // next always enqueues 1 prefetch task before it takes away an item
                // need to leave one space for that
                if (!prefetch_unsafe()) break;
            }
            //LOG(INFO) << "Starting " << nth << " threads.";
            CHECK(threads.empty());
            for (unsigned i = 0; i < nth; ++i) {
                threads.emplace_back([this](){this->worker();});
            }
        }

        void stop () {
            CHECK(started);
            started = false;
            eos = true;
            has_pending.notify_all();
            for (auto &th: threads) {
                th.join();
            }
            threads.clear();
        }

    public:
        struct Config: public Stream::Config, public Loader::Config {
            bool cache;
            unsigned preload;
            unsigned threads;   // 0 to use all cores
            Config (): cache(true),
                preload(DEFAULT_PRELOAD),
                threads(0)
            {
            }
        };

        PrefetchStream (fs::path const &p, Config const &c) 
            : Stream(p, c), Loader(c), started(false),
            nth(c.threads > 0? c.threads: DEFAULT_THREADS), queue(c.preload+1) {
            if (c.cache) {
                cache.resize(total());
            }
            // enqueue tasks
            start();
        }

        ~PrefetchStream () {
            stop();
        }

        void reset () {
            stop();
            Stream::reset();
            start();
        }

        Value &peek () {
            unique_lock lock(mutex);
            if (!peeked) {
                prefetch_unsafe();
                peeked = true;
            }
            Task &next = queue[next_loaded];
            while (next.status != Task::LOADED) {
                if (inqueue == 0) {
                    throw EoS();
                }
                has_loaded.wait(lock);
            }
            return next.value;
        }

        Value &&next () {
            unique_lock lock(mutex);
            if (peeked) {
                peeked = false;
            }
            else {
                prefetch_unsafe();
            }
            Task &next = queue[next_loaded];
            while (next.status != Task::LOADED) {
                if (inqueue == 0) {
                    throw EoS();
                }
                has_loaded.wait(lock);
            }
            value_holder.swap(next.value);
            next.status = Task::EMPTY;
            --inqueue;
            next_loaded = (next_loaded + 1) % queue.size();
            return std::move(value_holder);
        }
    };
}
