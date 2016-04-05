#include <linux/types.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <set>
#include <boost/filesystem/fstream.hpp>
#include "picpac.h"

namespace picpac {


    Record::Record (float label, fs::path const &image) {
        uintmax_t sz = fs::file_size(image);
        if (sz == static_cast<uintmax_t>(-1)) throw BadFile(image);
        alloc(label, sz);
        //meta->fields[0].type = FIELD_FILE;
        fs::ifstream is(image, std::ios::binary);
        is.read(field_ptrs[0], meta_ptr->fields[0].size);
        if (!is) throw BadFile(image);
    }

    Record::Record (float label, string const &image) {
        alloc(label, image.size());
        std::copy(image.begin(), image.end(), field_ptrs[0]);
    }

    Record::Record (float label, fs::path const &image, string const &extra) {
        uintmax_t sz = fs::file_size(image);
        if (sz == static_cast<uintmax_t>(-1)) throw BadFile(image);
        alloc(label, sz, extra.size());
        fs::ifstream is(image, std::ios::binary);
        //meta->fields[0].type = FIELD_FILE;
        is.read(field_ptrs[0], meta_ptr->fields[0].size);
        if (!is) throw BadFile(image);
        //meta->fields[1].type = FIELD_TEXT;
        std::copy(extra.begin(), extra.end(), field_ptrs[1]);
    }

    Record::Record (float label, fs::path const &image, fs::path const &image2) {
        uintmax_t sz = fs::file_size(image);
        if (sz == static_cast<uintmax_t>(-1)) throw BadFile(image);
        uintmax_t sz2 = fs::file_size(image2);
        if (sz2 == static_cast<uintmax_t>(-1)) throw BadFile(image2);
        alloc(label, sz, sz2);
        fs::ifstream is(image, std::ios::binary);
        //meta->fields[0].type = FIELD_FILE;
        is.read(field_ptrs[0], meta_ptr->fields[0].size);
        if (!is) throw BadFile(image);
        //meta->fields[1].type = FIELD_TEXT;
        fs::ifstream is2(image2, std::ios::binary);
        //meta->fields[0].type = FIELD_FILE;
        is2.read(field_ptrs[1], meta_ptr->fields[1].size);
        if (!is2) throw BadFile(image);
    }

    Record::Record (float label, string const &image, string const &extra) {
        alloc(label, image.size(), extra.size());
        std::copy(image.begin(), image.end(), field_ptrs[0]);
        std::copy(extra.begin(), extra.end(), field_ptrs[1]);
    }

#define CHECK_OFFSET    1
    ssize_t Record::write (int fd) const {
#ifdef CHECK_OFFSET
        off_t off = lseek(fd, 0, SEEK_CUR);
        CHECK(off >= 0);
        CHECK(off % RECORD_ALIGN == 0);
        off_t begin = off;
#endif
        ssize_t written = 0;
        ssize_t r = ::write(fd, &data[0], data.size());
        if (r != ssize_t(data.size())) return -1;
        written += r;
        ssize_t roundup = (written + RECORD_ALIGN - 1) / RECORD_ALIGN * RECORD_ALIGN;
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
        if (sz != ssize_t(size)) return -1;
        meta_ptr = reinterpret_cast<Meta *>(&data[0]);
        unsigned o = sizeof(Meta);
        for (unsigned i = 0; i < meta_ptr->width; ++i) {
            if (o >= size) throw DataCorruption();
            field_ptrs[i] = &data[o];
            o += meta_ptr->fields[i].size;
        }
        if (o > size) throw DataCorruption();
        return sz;
    }

    FileWriter::FileWriter (fs::path const &path) {
        fd = open(path.native().c_str(), O_CREAT | O_EXCL | O_WRONLY, 0666);
        CHECK(fd >= 0) << "fail to open " << path;
        open_segment();
    }

    FileWriter::~FileWriter () {
        off_t off = lseek(fd, 0, SEEK_CUR);
        //std::cerr << "CLOSE: " << off << std::endl;
        close_segment();
        int r = ftruncate(fd, off);
        CHECK(r == 0);
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
        seg.labels[next] = r.meta().label;
        seg.sizes[next++] = sz;
    }

    FileReader::FileReader (fs::path const &path) {
        fd = open(path.native().c_str(), O_RDONLY);
        CHECK(fd >= 0);
    }

    FileReader::~FileReader () {
        close(fd);
    }

    void FileReader::ping (vector<Locator> *l) {
        l->clear();
        struct stat st;
        int r = fstat(fd, &st);
        CHECK(r == 0);
        uint64_t off = 0;
        SegmentHeader seg;
        uint32_t s = 0;
        while (int64_t(off) < st.st_size) {
            /*
            uint64_t x = lseek(fd, off, SEEK_SET);
            CHECK(x == off);
            */
            ssize_t rd = ::pread(fd, reinterpret_cast<char *>(&seg), sizeof(seg), off);
            CHECK(rd == sizeof(seg));
            off += sizeof(seg);
            // append seg entries to list
            for (unsigned i = 0; i < seg.size; ++i) {
                Locator e;
                e.label = seg.labels[i];
                e.offset = off;
                e.size = seg.sizes[i];
                e.serial = s++;
                l->push_back(e);
                off += seg.sizes[i];
            }
            CHECK(off == seg.link);
            off = seg.link;
        }
    }

    void check_sort_dedupe_keys (unsigned splits, vector<unsigned> *keys) {
        std::sort(keys->begin(), keys->end());
        keys->resize(std::unique(keys->begin(), keys->end()) - keys->begin());
        CHECK(keys->size());
        for (unsigned k: *keys) {
            CHECK(k < splits);
        }
    }

    Stream::Stream (fs::path const &path, Config const &c)
        : FileReader(path), config(c), rng(config.seed), next_group(0)
    {
        vector<Locator> all;
        ping(&all);
        sz_total = all.size();
        if (config.stratify) {
            vector<vector<Locator>> C(MAX_CATEGORIES);
            int nc = 0;
            for (auto e: all) {
                int c = int(e.label);
                CHECK(c == e.label) << "We cannot stratify float labels.";
                CHECK(c >= 0) << "We cannot stratify label -1.";
                CHECK(c < int(MAX_CATEGORIES)) << "Too many categories (2000 max): " << c;
                C[c].push_back(e);
                if (c > nc) nc = c;
            }
            ++nc;
            groups.resize(nc);
            for (int c = 0; c < nc; ++c) {
                groups[c].id = c;
                groups[c].next = 0;
                groups[c].index.swap(C[c]);
            }
        }
        else {
            groups.resize(1);
            groups[0].id = 0;
            groups[0].next = 0;
            groups[0].index.swap(all);
        }
        CHECK(groups.size());
        if (config.shuffle) {
            for (auto &g: groups) {
                std::shuffle(g.index.begin(), g.index.end(), rng);
            }
        }
        unsigned K = config.split;
        vector<unsigned> keys;
        if (config.split_keys.size()) {
            CHECK(config.split_fold < 0) << "Cannot use keys and fold simultaneously.";
            keys = config.split_keys;
            check_sort_dedupe_keys(config.split, &keys);
        }
        else {
            // setup k-fold cross validation
            for (unsigned k = 0; k < K; ++k) {
                if (k != config.split_fold) keys.push_back(k);
            }
        }
        if (config.split_negate) {
            std::set<unsigned> excl(keys.begin(), keys.end());
            keys.clear();
            for (unsigned k = 0; k < K; ++k) {
                if (excl.count(k) == 0) {
                    keys.push_back(k);
                }
            }
        }

        if (K > 1) for (auto &g: groups) {
            vector<Locator> picked;
            for (unsigned k: keys) {
                auto begin = g.index.begin() + (g.index.size() * k / K);
                auto end = g.index.begin() + (g.index.size() * (k + 1) / K);
                picked.insert(picked.end(), begin, end);
            }
            g.index.swap(picked);
            if (picked.empty()) {
                LOG(WARNING) << "empty group " << g.id;
            }
        }
        sz_used = 0;
        for (auto const &g: groups) {
            sz_used += g.index.size();
        }
        //LOG(INFO) << "using " << sz_used << " out of " << sz_total << " items in " << groups.size() << " groups.";
        reset();
    }

    Locator Stream::next ()  {
        // check next group
        // find the next non-empty group
        Locator e;
        for (;;) {
            // we scan C times
            // if there's a non-empty group, 
            // we must be able to find it within C times
            if (next_group >= group_index.size()) {
                if (group_index.empty()) throw EoS();
                next_group = 0;
            }
            auto &g = groups[group_index[next_group]];
            if (g.next >= g.index.size()) {
                if (config.loop) {
                    g.next = 0;
                    if (config.reshuffle) {
                        std::shuffle(g.index.begin(), g.index.end(), rng);
                    }
                }
                if (g.next >= g.index.size()) {
                    // must check again to cover two cases:
                    // 1. not loop
                    // 2. loop, but group is empty
                    // remove this group
                    for (unsigned x = next_group + 1; x < groups.size(); ++x) {
                        group_index[x-1] = group_index[x];
                    }
                    group_index.pop_back();
                    // we need to scan for next usable group
                    continue;
                }
            }
            CHECK(g.next < g.index.size());
            //std::cerr << g.id << '\t' << g.next << '\t' << groups.size() << std::endl;
            e = g.index[g.next];
            ++g.next;
            ++next_group;
            break;
        }
        return e;
    }
}

