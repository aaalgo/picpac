#include <linux/types.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <boost/filesystem/fstream.hpp>
#include "picpac.h"

namespace picpac {


    void Record::pack (double label, fs::path const &image) {
        uintmax_t sz = fs::file_size(image);
        if (sz == static_cast<uintmax_t>(-1)) throw BadFile(image);
        alloc(label, sz);
        //meta->fields[0].type = FIELD_FILE;
        fs::ifstream is(image, std::ios::binary);
        is.read(fields[0], meta->fields[0].size);
        if (!is) throw BadFile(image);
    }

    void Record::pack (double label, fs::path const &image, string const &extra) {
        uintmax_t sz = fs::file_size(image);
        if (sz == static_cast<uintmax_t>(-1)) throw BadFile(image);
        alloc(label, sz, extra.size());
        fs::ifstream is(image, std::ios::binary);
        //meta->fields[0].type = FIELD_FILE;
        is.read(fields[0], meta->fields[0].size);
        if (!is) throw BadFile(image);
        //meta->fields[1].type = FIELD_TEXT;
        std::copy(extra.begin(), extra.end(), fields[1]);
    }

#define CHECK_OFFSET    1
    ssize_t Record::write (int fd) const {
#ifdef CHECK_OFFSET
        off_t off = lseek(fd, 0, SEEK_CUR);
        CHECK(off >= 0);
        CHECK(off % RECORD_ALIGN == 0);
        off_t begin = off;
#endif
        size_t written = 0;
        ssize_t r = ::write(fd, &data[0], data.size());
        if (r != data.size()) return -1;
        written += r;
        size_t roundup = (written + RECORD_ALIGN - 1) / RECORD_ALIGN * RECORD_ALIGN;
        if (roundup > written) {
            off_t x = lseek(fd, (roundup - written), SEEK_CUR);
            CHECK(x > 0);
            written = roundup;
        }
#ifdef CHECK_OFFSET
        off = lseek(fd, 0, SEEK_CUR);
        CHECK(off - begin == written);
        CHECK(off % RECORD_ALIGN == 0);
#endif
        return written;
    }

    ssize_t Record::read (int fd, off_t off, size_t size) {
        data.resize(size);
        ssize_t sz = pread(fd, &data[0], size, off);
        if (sz != size) return -1;
        meta = reinterpret_cast<Meta *>(&data[0]);
        unsigned o = sizeof(Meta);
        for (unsigned i = 0; i < meta->width; ++i) {
            fields[i] = &data[o];
            o += meta->fields[i].size;
        }
        return sz;
    }

    FileWriter::FileWriter (fs::path const &path) {
        fd = open(path.native().c_str(), O_CREAT | O_EXCL | O_WRONLY, 0666);
        CHECK(fd >= 0) << "fail to open " << path;
        open_segment();
    }

    FileWriter::~FileWriter () {
        close_segment();
        close(fd);
    }

    void FileWriter::open_segment () {
        seg_off = lseek(fd, 0, SEEK_CUR);
        CHECK(seg_off >= 0);
        CHECK(seg_off % RECORD_ALIGN == 0);
        seg.init();
        ssize_t r = write(fd, reinterpret_cast<char const *>(&seg), sizeof(seg));
        CHECK_EQ(r, sizeof(seg));
        next = 0;
    }

    void FileWriter::close_segment () {
        off_t off = lseek(fd, 0, SEEK_CUR);
        CHECK(off >= 0);
        CHECK(off % RECORD_ALIGN == 0);
        seg.link = off;
        ssize_t r = pwrite(fd, reinterpret_cast<char const *>(&seg), sizeof(seg), seg_off);
        CHECK_EQ(r, sizeof(seg));
    }

    void FileWriter::append (Record const &r) {
        if (next >= MAX_SEG_RECORDS) {
            close_segment();
            open_segment();
        }
        ssize_t sz = r.write(fd);
        CHECK(sz > 0);
        ++seg.size;
        seg.labels[next] = r.meta->label;
        seg.sizes[next++] = sz;
    }

    FileReader::FileReader (fs::path const &path) {
        fd = open(path.native().c_str(), O_RDONLY);
        CHECK(fd >= 0);
        struct stat st;
        int r = fstat(fd, &st);
        CHECK(r == 0);
        uint64_t off = 0;
        SegmentHeader seg;
        while (off < st.st_size) {
            uint64_t x = lseek(fd, off, SEEK_SET);
            CHECK(x == off);
            ssize_t rd = ::read(fd, reinterpret_cast<char *>(&seg), sizeof(seg));
            CHECK(rd == sizeof(seg));
            off += sizeof(seg);
            // append seg entries to list
            for (unsigned i = 0; i < seg.size; ++i) {
                FileEntry e;
                e.label = seg.labels[i];
                e.offset = off;
                e.size = seg.sizes[i];
                push_back(e);
                off += seg.sizes[i];
            }
            CHECK(off == seg.link);
            off = seg.link;
        }
    }

    FileReader::~FileReader () {
        close(fd);
    }
}

