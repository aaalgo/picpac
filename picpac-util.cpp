#include <sstream>
#include <boost/assert.hpp>
#include "picpac-util.h"

namespace picpac {

    bool is_url (std::string const &url) {
        if (url.compare(0, 7, "http://") == 0) return true;
        if (url.compare(0, 8, "https://") == 0) return true;
        if (url.compare(0, 6, "ftp://") == 0) return true;
        return false;
    }   

    string CachedDownloader::DEFAULT_AGENT("Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1");

    fs::path CachedDownloader::download (const std::string &url) {
        if (!is_url(url)) return fs::path(url);
        fs::path path = cache_path(url);
        if (fs::exists(path)) {
            return path;
        }
        std::ostringstream ss;
        ss << "wget --output-document=" << path.native() << " --quiet -nv --no-check-certificate";
        if (retry > 0) {
            ss << " --tries=" << retry;
        }
        if (timeout > 0) {
            ss << " --timeout=" << timeout;
        }
        if (agent.size()) {
            ss << " --user-agent=\"" << agent << "\"";
        }
        ss << ' ' << '"' << url << '"';
        string cmd = ss.str();
        LOG(INFO) << cmd;
        int r = ::system(cmd.c_str());
        CHECK(r == 0);
        return path;
    }

    namespace from_boost_uuid_detail {

    BOOST_STATIC_ASSERT(sizeof(unsigned char)*8 == 8);
    BOOST_STATIC_ASSERT(sizeof(unsigned int)*8 == 32);

    inline unsigned int left_rotate(unsigned int x, std::size_t n)
    {
        return (x<<n) ^ (x>> (32-n));
    }

    class sha1
    {
    public:
        typedef unsigned int(&digest_type)[5];
    public:
        sha1();

        void reset();

        void process_byte(unsigned char byte);
        void process_block(void const* bytes_begin, void const* bytes_end);
        void process_bytes(void const* buffer, std::size_t byte_count);

        void get_digest(digest_type digest);

    private:
        void process_block();
        void process_byte_impl(unsigned char byte);

    private:
        unsigned int h_[5];

        unsigned char block_[64];

        std::size_t block_byte_index_;
        std::size_t bit_count_low;
        std::size_t bit_count_high;
    };

    inline sha1::sha1()
    {
        reset();
    }

    inline void sha1::reset()
    {
        h_[0] = 0x67452301;
        h_[1] = 0xEFCDAB89;
        h_[2] = 0x98BADCFE;
        h_[3] = 0x10325476;
        h_[4] = 0xC3D2E1F0;

        block_byte_index_ = 0;
        bit_count_low = 0;
        bit_count_high = 0;
    }

    inline void sha1::process_byte(unsigned char byte)
    {
        process_byte_impl(byte);

        if (bit_count_low < 0xFFFFFFF8) {
            bit_count_low += 8;
        } else {
            bit_count_low = 0;

            if (bit_count_high <= 0xFFFFFFFE) {
                ++bit_count_high;
            } else {
                BOOST_THROW_EXCEPTION(std::runtime_error("sha1 too many bytes"));
            }
        }
    }

    inline void sha1::process_byte_impl(unsigned char byte)
    {
        block_[block_byte_index_++] = byte;

        if (block_byte_index_ == 64) {
            block_byte_index_ = 0;
            process_block();
        }
    }

    inline void sha1::process_block(void const* bytes_begin, void const* bytes_end)
    {
        unsigned char const* begin = static_cast<unsigned char const*>(bytes_begin);
        unsigned char const* end = static_cast<unsigned char const*>(bytes_end);
        for(; begin != end; ++begin) {
            process_byte(*begin);
        }
    }

    inline void sha1::process_bytes(void const* buffer, std::size_t byte_count)
    {
        unsigned char const* b = static_cast<unsigned char const*>(buffer);
        process_block(b, b+byte_count);
    }

    inline void sha1::process_block()
    {
        unsigned int w[80];
        for (std::size_t i=0; i<16; ++i) {
            w[i]  = (block_[i*4 + 0] << 24);
            w[i] |= (block_[i*4 + 1] << 16);
            w[i] |= (block_[i*4 + 2] << 8);
            w[i] |= (block_[i*4 + 3]);
        }
        for (std::size_t i=16; i<80; ++i) {
            w[i] = left_rotate((w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]), 1);
        }

        unsigned int a = h_[0];
        unsigned int b = h_[1];
        unsigned int c = h_[2];
        unsigned int d = h_[3];
        unsigned int e = h_[4];

        for (std::size_t i=0; i<80; ++i) {
            unsigned int f;
            unsigned int k;

            if (i<20) {
                f = (b & c) | (~b & d);
                k = 0x5A827999;
            } else if (i<40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (i<60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }

            unsigned temp = left_rotate(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = left_rotate(b, 30);
            b = a;
            a = temp;
        }

        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
    }

    inline void sha1::get_digest(digest_type digest)
    {
        // append the bit '1' to the message
        process_byte_impl(0x80);

        // append k bits '0', where k is the minimum number >= 0
        // such that the resulting message length is congruent to 56 (mod 64)
        // check if there is enough space for padding and bit_count
        if (block_byte_index_ > 56) {
            // finish this block
            while (block_byte_index_ != 0) {
                process_byte_impl(0);
            }

            // one more block
            while (block_byte_index_ < 56) {
                process_byte_impl(0);
            }
        } else {
            while (block_byte_index_ < 56) {
                process_byte_impl(0);
            }
        }

        // append length of message (before pre-processing) 
        // as a 64-bit big-endian integer
        process_byte_impl( static_cast<unsigned char>((bit_count_high>>24) & 0xFF) );
        process_byte_impl( static_cast<unsigned char>((bit_count_high>>16) & 0xFF) );
        process_byte_impl( static_cast<unsigned char>((bit_count_high>>8 ) & 0xFF) );
        process_byte_impl( static_cast<unsigned char>((bit_count_high)     & 0xFF) );
        process_byte_impl( static_cast<unsigned char>((bit_count_low>>24) & 0xFF) );
        process_byte_impl( static_cast<unsigned char>((bit_count_low>>16) & 0xFF) );
        process_byte_impl( static_cast<unsigned char>((bit_count_low>>8 ) & 0xFF) );
        process_byte_impl( static_cast<unsigned char>((bit_count_low)     & 0xFF) );

        // get final digest
        digest[0] = h_[0];
        digest[1] = h_[1];
        digest[2] = h_[2];
        digest[3] = h_[3];
        digest[4] = h_[4];
    }
    }

    void sha1sum (char const *data, unsigned length, std::string *checksum) {
        uint32_t digest[5];
        from_boost_uuid_detail::sha1 sha1;
        sha1.process_block(data, data+length);
        sha1.get_digest(digest);
        static char const digits[] = "0123456789abcdef";
        checksum->clear();
        for(uint32_t c: digest) {
            checksum->push_back(digits[(c >> 28) & 0xF]);
            checksum->push_back(digits[(c >> 24) & 0xF]);
            checksum->push_back(digits[(c >> 20) & 0xF]);
            checksum->push_back(digits[(c >> 16) & 0xF]);
            checksum->push_back(digits[(c >> 12) & 0xF]);
            checksum->push_back(digits[(c >> 8) & 0xF]);
            checksum->push_back(digits[(c >> 4) & 0xF]);
            checksum->push_back(digits[c & 0xF]);
        }
    }
}
