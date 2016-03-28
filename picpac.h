#ifndef PICPAC_INCLUDE
#define PICPAC_INCLUDE
#include <array>
#include <vector>
#include <string>
#include <mutex>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <glog/logging.h>

namespace picpac {

    using std::array;
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

    static constexpr int MAX_CATEGORIES = (1 << 23);

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

    struct FileEntry {
        off_t offset;
        uint32_t size;
        float label;
    };

    class FileReader:public vector<FileEntry> {
        int fd;
    public:
        FileReader (fs::path const &path);
        ~FileReader ();
        void read (unsigned n, Record *r) {
            auto const &e = at(n);
            r->read(fd, e.offset, e.size);
        }
    };
}

#endif
