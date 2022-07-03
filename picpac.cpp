#include <execinfo.h>
#include <iostream>
#include <linux/types.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <set>
#include <boost/filesystem/fstream.hpp>
#include "picpac.h"

namespace picpac {

    Stack::Stack (): symbols(nullptr) {
        void * stack[MAX_BACKTRACE];
        int sz = backtrace(stack, MAX_BACKTRACE);
        symbols = backtrace_symbols(stack, sz);
        resize(sz);
        for (int i = 0; i < sz; ++i) {
            at(i) = symbols[i];
        }
    }

    Stack::~Stack () {
        if (symbols) {
            free(symbols);
        }
    }   

    std::string Stack::format (std::string const &prefix) const {
        std::ostringstream ss; 
        ss << "backtrace" << std::endl;
        for (unsigned i = 0; i < this->size(); ++i) {
            ss << prefix << '#' << i << "  " << this->at(i) << std::endl;
        }
        return ss.str();
    }   


    Record::Record (float label, fs::path const &image) {
        uintmax_t sz = fs::file_size(image);
        if (sz == static_cast<uintmax_t>(-1)) throw BadFile(image);
        alloc(label, sz);
        //meta->fields[0].type = FIELD_FILE;
        ifstream is(image, std::ios::binary);
        is.read(field_ptrs[0], meta_ptr->fields[0].size);
        if (!is) throw BadFile(image);
    }

    Record::Record (float label, string const &image) {
        alloc(label, image.size());
        std::copy(image.begin(), image.end(), field_ptrs[0]);
    }

    Record::Record (float label, string_view buf) {
        size_t sz = buf.size();
        alloc(label, sz);
        char const *begin = buf.data();
        std::copy(begin, begin + sz, field_ptrs[0]);
    }

    Record::Record (float label, fs::path const &image, string const &extra) {
        uintmax_t sz = fs::file_size(image);
        if (sz == static_cast<uintmax_t>(-1)) throw BadFile(image);
        alloc(label, sz, extra.size());
        ifstream is(image, std::ios::binary);
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
        ifstream is(image, std::ios::binary);
        //meta->fields[0].type = FIELD_FILE;
        is.read(field_ptrs[0], meta_ptr->fields[0].size);
        if (!is) throw BadFile(image);
        //meta->fields[1].type = FIELD_TEXT;
        ifstream is2(image2, std::ios::binary);
        //meta->fields[0].type = FIELD_FILE;
        is2.read(field_ptrs[1], meta_ptr->fields[1].size);
        if (!is2) throw BadFile(image);
    }

    Record::Record (float label, string const &image, string const &extra) {
        alloc(label, image.size(), extra.size());
        std::copy(image.begin(), image.end(), field_ptrs[0]);
        std::copy(extra.begin(), extra.end(), field_ptrs[1]);
    }

    Record::Record (float label, string_view buf, string_view buf2) {
        size_t sz = buf.size();
        size_t sz2 = buf2.size();
        alloc(label, sz, sz2);
        char const *begin = buf.data();
        char const *begin2 = buf2.data();
        std::copy(begin, begin + sz, field_ptrs[0]);
        std::copy(begin2, begin2 + sz2, field_ptrs[1]);
    }

    Record::Record (float label, string_view buf, string_view buf2, string_view buf3) {
        size_t sz = buf.size();
        size_t sz2 = buf2.size();
        size_t sz3 = buf3.size();
        alloc(label, sz, sz2, sz3);
        char const *begin = buf.data();
        char const *begin2 = buf2.data();
        char const *begin3 = buf3.data();
        std::copy(begin, begin + sz, field_ptrs[0]);
        std::copy(begin2, begin2 + sz2, field_ptrs[1]);
        std::copy(begin3, begin3 + sz3, field_ptrs[2]);
    }

    Record::Record (float label, string_view buf, string_view buf2, string_view buf3, string_view buf4) {
        size_t sz = buf.size();
        size_t sz2 = buf2.size();
        size_t sz3 = buf3.size();
        size_t sz4 = buf4.size();
        alloc(label, sz, sz2, sz3, sz4);
        char const *begin = buf.data();
        char const *begin2 = buf2.data();
        char const *begin3 = buf3.data();
        char const *begin4 = buf4.data();
        std::copy(begin, begin + sz, field_ptrs[0]);
        std::copy(begin2, begin2 + sz2, field_ptrs[1]);
        std::copy(begin3, begin3 + sz3, field_ptrs[2]);
        std::copy(begin4, begin4 + sz4, field_ptrs[3]);
    }

    Record::Record (float label, string const &image, string const &extra, string const &extra2) {
        alloc(label, image.size(), extra.size(), extra2.size());
        std::copy(image.begin(), image.end(), field_ptrs[0]);
        std::copy(extra.begin(), extra.end(), field_ptrs[1]);
        std::copy(extra2.begin(), extra2.end(), field_ptrs[2]);
    }

    Record::Record (float label, string const &image, string const &extra, string const &extra2, string const &extra3) {
        alloc(label, image.size(), extra.size(), extra2.size(), extra3.size());
        std::copy(image.begin(), image.end(), field_ptrs[0]);
        std::copy(extra.begin(), extra.end(), field_ptrs[1]);
        std::copy(extra2.begin(), extra2.end(), field_ptrs[2]);
        std::copy(extra3.begin(), extra3.end(), field_ptrs[3]);
    }

    Record::Record (float label, string const &image, string const &extra, string const &extra2, string const &extra3, string const &extra4) {
        alloc(label, image.size(), extra.size(), extra2.size(), extra3.size(), extra4.size());
        std::copy(image.begin(), image.end(), field_ptrs[0]);
        std::copy(extra.begin(), extra.end(), field_ptrs[1]);
        std::copy(extra2.begin(), extra2.end(), field_ptrs[2]);
        std::copy(extra3.begin(), extra3.end(), field_ptrs[3]);
        std::copy(extra4.begin(), extra4.end(), field_ptrs[4]);
    }

#define CHECK_OFFSET    1
    ssize_t Record::write (int fd, bool compact) const {
#ifdef CHECK_OFFSET
        off_t off = lseek(fd, 0, SEEK_CUR);
        CHECK(off >= 0);
        CHECK(compact || (off % RECORD_ALIGN == 0));
        off_t begin = off;
#endif
        ssize_t written = 0;
        ssize_t r = ::write(fd, &data[0], data.size());
        if (r != ssize_t(data.size())) return -1;
        written += r;
        ssize_t roundup = (written + RECORD_ALIGN - 1) / RECORD_ALIGN * RECORD_ALIGN;
        if ((!compact) && (roundup > written)) {
            off_t x = lseek(fd, (roundup - written), SEEK_CUR);
            CHECK(x > 0);
            written = roundup;
        }
#ifdef CHECK_OFFSET
        off = lseek(fd, 0, SEEK_CUR);
        CHECK(off - begin == written);
        CHECK(compact || (off % RECORD_ALIGN == 0));
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
        data.resize(o);
        return size;
    }

    void Record::replace (unsigned f, string const &buf, int type) {
        CHECK(f < meta_ptr->width);
        // calculate new size
        size_t sz = data.size() + buf.size() - meta_ptr->fields[f].size;
        string new_data;
        new_data.resize(sz);
        Meta *new_meta = reinterpret_cast<Meta *>(&new_data[0]);
        *new_meta = *meta_ptr;
        new_meta->fields[f].size = buf.size();
        if (type >= 0) {
            new_meta->fields[f].type = type;
        }
        // copy data
        char *out_ptr = &new_data[sizeof(Meta)];
        for (unsigned i = 0; i < meta_ptr->width; ++i) {
            char *in_ptr = field_ptrs[i];
            field_ptrs[i] = out_ptr;
            if (i == f) {   // copy from buffer
                std::copy(buf.begin(), buf.end(), out_ptr);
                out_ptr += buf.size();
            }
            else {
                size_t sz = meta_ptr->fields[i].size;
                std::copy(in_ptr, in_ptr + sz, out_ptr);
                out_ptr += sz;
            }
        }
        CHECK(unsigned(out_ptr - &new_data[0]) == new_data.size());
        data.swap(new_data);
        meta_ptr = new_meta;
        CHECK((meta_ptr == reinterpret_cast<Meta *>(&data[0]))
              && (field_ptrs[0] == &data[sizeof(Meta)]));
                //<< "C++ string::swap is not preserving memory";
    }

    FileWriter::FileWriter (fs::path const &path, int flags_): flags(flags_) {
        int f = O_WRONLY | O_CREAT;
        if (flags_ & OVERWRITE) {
            f |= O_TRUNC;
        }
        else {
            f |= O_EXCL;
        }
        fd = open(path.native().c_str(), f, 0666);
        CHECK(fd >= 0); // << "fail to open " << path;
        open_segment();
    }

    FileWriter::~FileWriter () {
        off_t off = lseek(fd, 0, SEEK_CUR);
        close_segment();
        int r = ftruncate(fd, off);
        CHECK(r == 0);
        close(fd);
    }

    void FileWriter::open_segment () {
        seg_off = lseek(fd, 0, SEEK_CUR);
        CHECK(seg_off >= 0);
        CHECK(compact() || (seg_off % RECORD_ALIGN == 0));
        seg.init();
        ssize_t r = write(fd, reinterpret_cast<char const *>(&seg), sizeof(seg));
        CHECK(r == sizeof(seg));
        next = 0;
    }

    void FileWriter::close_segment () {
        off_t off = lseek(fd, 0, SEEK_CUR);
        CHECK(off >= 0);
        CHECK(compact() || (off % RECORD_ALIGN == 0));
        seg.link = off;
        ssize_t r = pwrite(fd, reinterpret_cast<char const *>(&seg), sizeof(seg), seg_off);
        CHECK(r == sizeof(seg));
    }

    void FileWriter::append (Record const &r) {
        if (next >= MAX_SEG_RECORDS) {
            close_segment();
            open_segment();
        }
        ssize_t sz = r.write(fd, compact());
        CHECK(sz > 0);
        ++seg.size;
        seg.groups[next] = (flags & INDEX_LABEL2) ? r.meta().label2 : r.meta().label;
        seg.sizes[next++] = sz;
    }

    FileReader::FileReader (fs::path const &path) {
        fd = open(path.native().c_str(), O_RDONLY);
        CHECK(fd >= 0);
    }

    FileReader::~FileReader () {
        close(fd);
    }

    void FileReader::ping (vector<Locator> *l, uint32_t file) {
        //l->clear();
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
                e.group = seg.groups[i];
                e.offset = off;
                e.size = seg.sizes[i];
                e.serial = s++;
                e.file = file;
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
        readers.push_back(this);
        if (config.mixin.size()) {
            readers.push_back(new FileReader(fs::path(config.mixin)));
            CHECK(readers.back());
        }
        vector<Locator> all;
        for (unsigned i = 0; i < readers.size(); ++i) {
            size_t before = all.size();
            readers[i]->ping(&all, i);
            if (i && config.mixin_max) {
                size_t bound = before + config.mixin_max;
                if (all.size() > bound) {
                    if (config.mixin_randomize) {
                        std::shuffle(all.begin() + before, all.end(), rng);
                    }
                    all.resize(bound);
                }
            }
        }
        vector<float> group_reset{-1, config.mixin_group_reset};
        vector<float> group_delta{0, config.mixin_group_delta};
        for (auto &l: all) {
            if (group_reset[l.file] >= 0) {
                l.group = group_reset[l.file];
            }
            l.group += group_delta[l.file];
        }
        sz_total = all.size();
        ncat = 0;
        for (auto const &e: all) {
            int c = int(e.group);
            if ((c != e.group) || (c < 0)) {
                ncat = 0;
                break;
            }
            if (c > int(ncat)) ncat = c;
        }
        // if group is float or group < 0
        // ncat is to be 0
        if (config.stratify && (ncat >= MAX_CATEGORIES)) {
            logging::error("Too many categories (2000 max): {}", ncat);
            ncat = 0;
        }

        if (ncat == 0) {
            logging::warn("Stratefication disabled.");
        }


        if (config.stratify && (ncat > 0)) {
            vector<vector<Locator>> C(ncat+1);
            int nc = 0;
            for (auto const &e: all) {
                int c;
                c = int(e.group);
                CHECK(c == e.group); // << "We cannot stratify float labels.";
                CHECK(c >= 0); // << "We cannot stratify label -1.";
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
        int K = config.split;
        vector<unsigned> keys;
        if (config.split_keys.size()) {
            if (config.split_fold >= 0) {
                logging::error("Cannot use keys and fold simultaneously, set split_fold to -1.");
                CHECK(0);
            }
            keys = config.split_keys;
            check_sort_dedupe_keys(config.split, &keys);
        }
        else {
            // setup k-fold cross validation
            for (int k = 0; k < K; ++k) {
                if (k != config.split_fold) keys.push_back(k);
            }
        }
        if (config.split_negate) {
            std::set<unsigned> excl(keys.begin(), keys.end());
            keys.clear();
            for (int k = 0; k < K; ++k) {
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
                logging::warn("empty group {}", g.id);
            }
        }

        if (!config.oversample) {
            vector<Locator> all;
            for (auto &g: groups) {
                all.insert(all.end(), g.index.begin(), g.index.end());
            }
            if (config.shuffle) {
                std::shuffle(all.begin(), all.end(), rng);
            }
            groups.resize(1);
            groups[0].id = 0;
            groups[0].next = 0;
            groups[0].index.swap(all);
        }

        sz_used = 0;
        for (auto const &g: groups) {
            sz_used += g.index.size();
        }
        //logging::info("using " << sz_used << " out of " << sz_total << " items in " << groups.size() << " groups.";
        reset();
    }

    Stream::~Stream () {
        for (auto p: readers) {
            if (p != this) delete p;
        }
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

    char const *EMPTY_BUFFER = "";
}

