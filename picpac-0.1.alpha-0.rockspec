package = "picpac"
version = "0.1.alpha-0"

source = {
   url = "./", ---"git://github.com/aaalgo/picpac",
   tag = "master"
}

description = {
   summary = "An image database and streamer for deep learning.",
   detailed = [[
   ]],
   homepage = "https://github.com/aaalgo/picpac",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
}

build = {
    type = "make",
    makefile = "Makefile.lua",
    build_variables = {
        CXXFLAGS="$(CFLAGS) -I$(LUA_INCDIR)",
        LDFLAGS="$(LIBFLAG) -L$(LUA_LIBDIR)",
    },
    install_variables = {
        LIBDIR="$(LIBDIR)",
    },
}
