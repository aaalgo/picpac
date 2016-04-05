#include <cxxabi.h>
#include <map>
#include "picpac-cv.h"

std::map<std::string, std::string> PROTO_MAP{
    {"std::string", "string"},
    {"unsigned int", "uint32"},
    {"int", "int32"},
};

template <class T>
std::string type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    auto it = PROTO_MAP.find(r);
    if (it != PROTO_MAP.end()) return it->second;
    return r;
}

using namespace std;
int main () {
    PICPAC_CONFIG conf;
    int cnt = 2;
    cout << "  required string path = 1;" << endl;
#define PICPAC_CONFIG_UPDATE(C,P) \
    cout << "  optional " << type_name<decltype(C.P)>() << ' ' << #P << " = " << (cnt++) << ';' << endl
    PICPAC_CONFIG_UPDATE_ALL(conf);
#undef PICPAC_CONFIG_UPDATE
    return 0;
}
