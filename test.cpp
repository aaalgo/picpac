#include <map>
#include <iostream>
#include "picpac.h"

using namespace std;
using namespace picpac;

int main (int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    unsigned C = 17;
    fs::path path("test.pac");
    map<unsigned, int> cnt;
    map<unsigned, int> cnt_id;
    {
        fs::remove(path);
        FileWriter out(path);
        for (unsigned i = 0; i < MAX_SEG_RECORDS * 3; ++i) {
            Record r(rand() % C, fs::path("picpac.h"), lexical_cast<string>(i));
            r.meta().id = i;
            cnt[r.meta().label] += 10;
            cnt_id[i] += 10;
            out.append(r);
        }
    }
    {
        auto sz = fs::file_size("picpac.h");
        FileReader in(path);
        vector<Locator> ll;
        in.ping(&ll);
        for (unsigned i = 0; i < ll.size(); ++i) {
            Record r;
            in.read(ll[i], &r);
            CHECK(r.meta().id == i);
            CHECK(r.meta().fields[0].size == sz);
            /*
            if (i == 10) {
                cout.write(r.fields[0], sz);
            }
            */
        }
    }
    for (auto p: cnt) {
        cout << p.first << ": " << p.second << endl;
    }
    for (unsigned F = 0; F < 5; ++F) {
        for (int train = 0; train < 2; ++train) {
            Stream::Config conf;
            conf.kfold(5, F, bool(train));
            conf.loop = false;
            Stream stream(path, conf);
            for (unsigned xx = 0; xx < 2; ++xx) {
                for (;;) {
                    try {
                        Record r;
                        stream.read_next(&r);
                        cnt[r.meta().label] -= 1;
                        cnt_id[r.meta().id] -= 1;
                    }
                    catch (EoS) {
                        break;
                    }
                }
                stream.reset();
            }
        }
    }
    for (auto p: cnt) {
        CHECK(p.second == 0);
    }
    for (auto p: cnt_id) {
        CHECK(p.second == 0);
    }
    cout << "XXX" << endl;
    for (int stra = 0; stra < 2; ++stra) {
        Stream::Config conf;
        conf.kfold(5, 3, true);
        conf.stratify = bool(stra);
        cout << "STR: " << conf.stratify << endl;
        Stream stream(path, conf);
        cnt.clear();
        for (unsigned i = 0; i < MAX_SEG_RECORDS * 10; ++i) {
            Record r;
            stream.read_next(&r);
//            cerr << r.meta->id << '\t' << r.meta->label << endl;
            cnt[r.meta().label] += 1;
        }
        for (auto p: cnt) {
            cout << p.first << ": " << p.second << endl;
        }
    }
}
