#pragma once
#include <fstream>

namespace picpac {
    class Tar {
        static const unsigned BLOCK_SIZE = 512;
    public:
        struct __attribute__((__packed__)) posix_header {
          char name[100];               /*   0 */
          char mode[8];                 /* 100 */
          char uid[8];                  /* 108 */
          char gid[8];                  /* 116 */
          char size[12];                /* 124 */
          char mtime[12];               /* 136 */
          char chksum[8];               /* 148 */
          char typeflag;                /* 156 */
          char linkname[100];           /* 157 */
          char magic[6];                /* 257 */
          char version[2];              /* 263 */
          char uname[32];               /* 265 */
          char gname[32];               /* 297 */
          char devmajor[8];             /* 329 */
          char devminor[8];             /* 337 */
          char prefix[155];             /* 345 */
          char padding[12];
        };
    private:
        static bool is_end (posix_header const *hdr) {
            char const *p = (char const *)hdr;
            for (unsigned i = 0; i < BLOCK_SIZE; ++i) {
                if (p[i]) return false;
            }
            return true;
        }
        std::ifstream is;
        posix_header header;
    public:
        Tar (string const &path): is(path.c_str(), std::ios::binary) {
        }

        bool next (string *data, posix_header const **phdr = nullptr) {
            is.read(reinterpret_cast<char *>(&header), BLOCK_SIZE);
            if (!is) return false;
            if (is_end(&header)) return false;
            unsigned sz = strtol(header.size, NULL, 8);
            unsigned skip = (sz + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE - sz;
            data->resize(sz);
            is.read(&data->at(0), sz);
            is.seekg(skip, std::ios::cur);
            if (phdr) {
                *phdr = &header;
            }
            return true;
        }
    
    };
}

