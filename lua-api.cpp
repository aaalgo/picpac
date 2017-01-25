#define LUA_LIB
extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
#include "luaT.h"
}
#include <iostream>
#include <TH/TH.h>
#include "picpac.h"
#include "picpac-cv.h"

using namespace std;
using namespace picpac;

namespace {

    template <typename T>
    struct Extractor {
        static T apply (lua_State *);
    };
    template <>
    struct Extractor <int32_t> {
        static int32_t apply (lua_State *L) {
            return luaL_checkinteger(L, -1);
        }
    };
    template <>
    struct Extractor <uint32_t> {
        static uint32_t apply (lua_State *L) {
            return luaL_checkinteger(L, -1);
        }
    };
    template <>
    struct Extractor <float> {
        static float apply (lua_State *L) {
            return luaL_checknumber(L, -1);
        }
    };
    template <>
    struct Extractor <bool> {
        static bool apply (lua_State *L) {
            return luaT_checkboolean(L, -1);
        }
    };
    template <>
    struct Extractor <string> {
        static string apply (lua_State *L) {
            return luaL_checkstring(L, -1);
        }
    };

    class LuaDict {
        lua_State *L;
        int idx;
    public:
        LuaDict (lua_State *L_, int idx_): L(L_), idx(idx_) {
        }
        template <typename T>
        T get (char const *name, T const &def) {
            lua_getfield(L, idx, name);
            T v;
            if (lua_isnil(L, -1)) {
                v = def;
            }
            else {
                v = Extractor<T>::apply(L);
            }
            //lua_pop(L, 1);
            return v;
        }
    };
    
    static char const *TNAME_MODULE = "picpac";
    static char const *TNAME_IMAGESTREAM = "picpac.ImageStream";
    static char const *TNAME_WRITER = "picpac.Writer";

    static int Writer_constructor (lua_State *L) {
        const char *db = luaL_checkstring(L, 1);
        FileWriter *ss = new FileWriter(fs::path(db), FileWriter::COMPACT);
        CHECK(ss) << "failed to construct Writer";
        luaT_pushudata(L, ss, TNAME_WRITER);
        return 1;
    }
    static FileWriter *checkWriter (lua_State *L, int n) {
        return *(FileWriter **)luaL_checkudata(L, n, TNAME_WRITER);
    }
    static int Writer_append (lua_State *L) {
        FileWriter *ss = checkWriter(L, 1);
        if (lua_isnumber(L, 1)) {
            Record record(float(luaL_checknumber(L, 1)),
                      string(luaL_checkstring(L, 2)));
            ss->append(record);
        }
        else {
            Record record(0, string(luaL_checkstring(L, 1)),
                      string(luaL_checkstring(L, 2)));
            ss->append(record);
        }
        return 0;
    }
    static int Writer_destructor (lua_State *L) {
        delete checkWriter(L, 1);
        return 0;
    }
    static int ImageStream_constructor (lua_State *L) {
        const char *db = luaL_checkstring(L, 1);
        luaL_checktype(L, 2, LUA_TTABLE);
        LuaDict dict(L, 2);
        BatchImageStream::Config config;
#define PICPAC_CONFIG_UPDATE(C, P) \
        C.P = dict.get<decltype(C.P)>(#P, C.P)
        PICPAC_CONFIG_UPDATE_ALL(config);
#undef PICPAC_CONFIG_UPDATE
        BatchImageStream *ss = new BatchImageStream(db, config);
        CHECK(ss) << "failed to construct BatchImageStream";
        luaT_pushudata(L, ss, TNAME_IMAGESTREAM);
        return 1;
    }
    static BatchImageStream *checkImageStream (lua_State *L, int n) {
        return *(BatchImageStream **)luaL_checkudata(L, n, TNAME_IMAGESTREAM);
    }
    static int ImageStream_size (lua_State *L) {
        BatchImageStream *ss = checkImageStream(L, 1);
        lua_pushinteger(L, lua_Integer(ss->size()));
        return 1;
    }
    static int ImageStream_next (lua_State *L) {
        BatchImageStream *ss = checkImageStream(L, 1);
        //cout << "ImageStream(" << ss << ")::next" << endl;
        vector<long> images_dims;
        vector<long> labels_dims;
        THFloatTensor *images = nullptr;
        THFloatTensor *labels = nullptr;
        try {
            ss->next_shape(&images_dims, &labels_dims);
            CHECK(images_dims.size() == 4);
            while (labels_dims.size() < 4) labels_dims.push_back(-1);
            images = THFloatTensor_newWithSize4d(images_dims[0],
                                        images_dims[1], images_dims[2], images_dims[3]);
            CHECK(THFloatTensor_isContiguous(images));
            labels = THFloatTensor_newWithSize4d(labels_dims[0],
                                        labels_dims[1], labels_dims[2], labels_dims[3]);
            CHECK(THFloatTensor_isContiguous(labels));
            float *images_buf = THFloatTensor_data(images);
            float *labels_buf = THFloatTensor_data(images);
            unsigned padding;
            ss->next_fill(images_buf, labels_buf, &padding);
            luaT_pushudata(L, images, "torch.FloatTensor");
            luaT_pushudata(L, labels, "torch.FloatTensor");
            luaT_pushinteger(L, lua_Integer(padding));
            return 3;
        }
        catch (EoS const &) {
            THFloatTensor_free(images);
            THFloatTensor_free(labels);
            lua_pushnil(L);
            return 1;
        }

    }
    static int ImageStream_destructor (lua_State *L) {
        delete checkImageStream(L, 1);
        return 0;
    }
}

extern "C" {
    LUALIB_API int luaopen_picpac (lua_State *L) {
        luaL_Reg PicPacMethods [] = {
            {"ImageStream", ImageStream_constructor},
            {"Writer", Writer_constructor},
            {NULL, NULL}
        };
        luaL_Reg WriterMethods [] = {
            {"append", Writer_append},
            {"__gc", Writer_destructor},
            {NULL, NULL}
        };
        luaL_Reg ImageStreamMethods [] = {
            {"next", ImageStream_next},
            {"size", ImageStream_size},
            {"__call", ImageStream_next},
            {"__gc", ImageStream_destructor},
            {NULL, NULL}
        };
        luaL_newmetatable(L, TNAME_WRITER);
        luaL_register(L, NULL, WriterMethods);
        lua_pushvalue(L, -1);
        lua_setfield(L, -1, "__index");
        luaL_newmetatable(L, TNAME_IMAGESTREAM);
        luaL_register(L, NULL, ImageStreamMethods);
        lua_pushvalue(L, -1);
        lua_setfield(L, -1, "__index");
        luaL_newmetatable(L, TNAME_MODULE);
        luaL_register(L, NULL, PicPacMethods);
        //luaL_setfuncs(L, NULL, ImageStreamMethods);
        lua_setglobal(L, "picpac");
        return 0;
    }
}
