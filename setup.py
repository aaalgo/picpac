#!/usr/bin/env python3
import sys
import os
import subprocess as sp
import numpy

def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    # parallel code
    N=8 # number of parallel compilations
    import multiprocessing.pool
    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool(N).map(_single_compile,objects))
    return objects
import distutils.ccompiler
distutils.ccompiler.CCompiler.compile=parallelCCompile
from distutils.core import setup, Extension


picpac = Extension('picpac_ts',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++17'], 
        include_dirs = ['/usr/local/include',
            '3rd/pybind11/include',
            '3rd/xtl/include',
            '3rd/xtensor/include',
            '3rd/xtensor-python/include',
            '3rd/xtensor-blas/include',
            '3rd/fmt/include',
            '3rd/spdlog/include'
            ],
        libraries = ['openblas'],
        library_dirs = ['/usr/local/lib', '.'],
        #sources = ['python-api.cpp', 'picpac.cpp', 'picpac-ts.cpp', 'transforms.cpp', 'json11/json11.cpp'],
        sources = ['python-api.cpp', 'picpac.cpp', "picpac-ts.cpp", 'transforms.cpp'],
        depends = ['picpac.h', 'picpac-ts.h'])

setup (name = 'picpac_ts',
       version = '0.2.2',
       url = 'https://github.com/aaalgo/picpac',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'BSD',
       description = 'This is a demo package',
       ext_modules = [picpac],
       )

