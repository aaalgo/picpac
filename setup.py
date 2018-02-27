import sys
import os
import cv2
import subprocess as sp
import numpy
from distutils.core import setup, Extension

libraries = []
cv2libs = sp.check_output('pkg-config --libs opencv', shell=True).decode('ascii')
cv_converter = 'pyboostcvconverter/src/pyboost_cv2_converter.cpp'
if 'opencv_imgcodecs' in cv2libs:
    libraries.append('opencv_imgcodecs')
    cv_converter = 'pyboostcvconverter/src/pyboost_cv3_converter.cpp'
    pass

numpy_include = os.path.join(os.path.abspath(os.path.dirname(numpy.__file__)), 'core', 'include')

if sys.version_info[0] < 3:
    boost_python = 'boost_python'
else:
    boost_python = 'boost_python-py%d%d' % (sys.version_info[0], sys.version_info[1])
    pass

libraries.extend(['opencv_highgui', 'opencv_imgproc', 'opencv_core', 'boost_filesystem', 'boost_system', boost_python, 'glog'])

picpac = Extension('picpac',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y'], 
        include_dirs = ['/usr/local/include', 'pyboostcvconverter/include', 'json11', numpy_include],
        libraries = libraries,
        library_dirs = ['/usr/local/lib'],
        sources = ['python-api.cpp', 'picpac.cpp', 'picpac-image.cpp', 'shapes.cpp', 'transforms.cpp', 'picpac-cv.cpp', 'json11/json11.cpp', cv_converter],
        depends = ['json11/json11.hpp', 'picpac.h', 'picpac-cv.h'])

picpac_legacy = Extension('_picpac',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y'], 
        include_dirs = ['/usr/local/include', 'json11', numpy_include],
        libraries = libraries,
        library_dirs = ['/usr/local/lib'],
        sources = ['legacy-python-api.cpp', 'picpac.cpp', 'picpac-cv.cpp', 'json11/json11.cpp'],
        depends = ['json11/json11.hpp', 'picpac.h', 'picpac-cv.h'])

setup (name = 'picpac',
       version = '0.2.2',
       url = 'https://github.com/aaalgo/picpac',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'BSD',
       description = 'This is a demo package',
       ext_modules = [picpac], #, picpac_legacy],
       py_modules = ['picpac_legacy.mxnet', 'picpac_legacy.neon'],
       requires = ["cv2"],
       )

