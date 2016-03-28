#include <iostream>
#include "picpac.h"

using namespace std;
using namespace picpac;

int main () {

    fs::path path("test.pac");
    {
        fs::remove(path);
        FileWriter out(path);
        for (unsigned i = 0; i < MAX_SEG_RECORDS * 3; ++i) {
            Record r;
            r.pack(i, fs::path("picpac.h"), lexical_cast<string>(i));
            out.append(r);
        }
    }
    {
        auto sz = fs::file_size("picpac.h");
        FileReader in(path);
        for (unsigned i = 0; i < in.size(); ++i) {
            CHECK(in[i].label == i);
            Record r;
            in.read(i, &r);
            CHECK(r.meta->fields[0].size == sz);
            if (i == 10) {
                cout.write(r.fields[0], sz);
            }
        }
    }

}
