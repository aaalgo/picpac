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
    static char const *TNAME_MODULE = "picpac_module";
    static char const *TNAME_CLASS = "picpac_class";
    static int ImageStream_constructor (lua_State *L) {
        const char *db = luaL_checkstring(L, 1);
        BatchImageStream **udata = (BatchImageStream **)lua_newuserdata(L, sizeof(BatchImageStream *));
        BatchImageStream::Config config;
        *udata = new BatchImageStream(db, config);
        CHECK(*udata) << "failed to construct BatchImageStream";
        luaL_getmetatable(L, TNAME_CLASS);
        lua_setmetatable(L, -2);
        return 1;
    }
    static BatchImageStream *checkImageStream (lua_State *L, int n) {
        return *(BatchImageStream **)luaL_checkudata(L, n, TNAME_CLASS);
    }
    static int ImageStream_size (lua_State *L) {
        BatchImageStream *ss = checkImageStream(L, 1);
        lua_pushinteger(L, lua_Integer(ss->size()));
        return 1;
    }
    static int ImageStream_next (lua_State *L) {
        BatchImageStream *ss = checkImageStream(L, 1);
        cout << "ImageStream(" << ss << ")::next" << endl;
        vector<long> images_dims;
        vector<long> labels_dims;
        ss->next_shape(&images_dims, &labels_dims);
        CHECK(images_dims.size() == 4);
        while (labels_dims.size() < 4) labels_dims.push_back(-1);
        THFloatTensor *images = THFloatTensor_newWithSize4d(images_dims[0],
                                    images_dims[1], images_dims[2], images_dims[3]);
        CHECK(THFloatTensor_isContiguous(images));
        THFloatTensor *labels = THFloatTensor_newWithSize4d(labels_dims[0],
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
    static int ImageStream_destructor (lua_State *L) {
        delete checkImageStream(L, 1);
        return 0;
    }
}

extern "C" {
    LUALIB_API int luaopen_picpac (lua_State *L) {
        luaL_Reg PicPacMethods [] = {
            {"ImageStream", ImageStream_constructor},
            {NULL, NULL}
        };
        luaL_Reg ImageStreamMethods [] = {
            {"next", ImageStream_next},
            {"size", ImageStream_size},
            {"__gc", ImageStream_destructor},
            {NULL, NULL}
        };
        luaL_newmetatable(L, TNAME_CLASS);
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
