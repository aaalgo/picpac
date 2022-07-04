#!/usr/bin/env python3
import sys
import os
import subprocess as sp
import numpy
from distutils.core import setup, Extension
if os.name != 'nt':
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

libraries = []
extra_include = []
for x in sp.check_output('pkg-config --cflags opencv4', shell=True).decode('ascii').strip().split(' '):
    assert x[:2] == '-I'
    extra_include.append(x[2:])
print(extra_include)

cv2libs = sp.check_output('pkg-config --libs opencv4', shell=True).decode('ascii')
if 'opencv_imgcodecs' in cv2libs:
    libraries.append('opencv_imgcodecs')
    pass

numpy_include = os.path.join(os.path.abspath(os.path.dirname(numpy.__file__)), 'core', 'include')

libraries.extend(['opencv_highgui', 'opencv_imgproc', 'opencv_core'])

picpac = Extension('picpac',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++17', '-Wno-terminate', '-Wno-sign-compare'],
        include_dirs = ['/usr/local/include', '3rd/pybind11_opencv_numpy', numpy_include, '3rd/spdlog/include', '3rd/fmt/include', '3rd/json/single_include/nlohmann', '3rd/pybind11/include'] + extra_include,
        libraries = libraries,
        library_dirs = ['/usr/local/lib'],
        sources = ['python-api.cpp', 'picpac.cpp', 'picpac-image.cpp', 'shapes.cpp', 'transforms.cpp', '3rd/pybind11_opencv_numpy/ndarray_converter.cpp'],
        depends = ['picpac.h', 'picpac-image.h', 'bachelor.h'])

#picpac_legacy = Extension('_picpac',
#        language = 'c++',
#        extra_compile_args = ['-O3', '-std=c++1y'], 
#        include_dirs = ['/usr/local/include', 'json11', numpy_include],
#        libraries = libraries,
#        library_dirs = ['/usr/local/lib'],
#        sources = ['legacy-python-api.cpp', 'picpac.cpp', 'picpac-cv.cpp', 'json11/json11.cpp'],
#        depends = ['json11/json11.hpp', 'picpac.h', 'picpac-cv.h'])

setup (name = 'picpac',
       version = '0.2.2',
       url = 'https://github.com/aaalgo/picpac',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'BSD',
       description = 'This is a demo package',
       ext_modules = [picpac], #, picpac_legacy],
       #py_modules = ['picpac_legacy.mxnet', 'picpac_legacy.neon'],
       #requires = ["cv2"],
       )

