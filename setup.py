import sys
import os
import numpy
import cv2
import subprocess as sp
from distutils.core import setup, Extension

libraries = []
cv2libs = sp.check_output('pkg-config --libs opencv', shell=True).decode('ascii')
if 'opencv_imgcodecs' in cv2libs:
    libraries.append('opencv_imgcodecs')
    pass

if sys.version_info[0] < 3:
    boost_python = 'boost_python'
else:
    boost_python = 'boost_python-py35'
    pass

libraries.extend(['opencv_highgui', 'opencv_imgproc', 'opencv_core', 'boost_filesystem', 'boost_system', boost_python, 'glog'])

numpy_root = os.path.join(os.path.abspath(os.path.dirname(numpy.__file__)), 'core')

picpac = Extension('picpac_ssd',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y'], 
        include_dirs = ['/usr/local/include', 'json11', os.path.join(numpy_root, 'include')],
        libraries = libraries,
        library_dirs = ['/usr/local/lib'],
        sources = ['python-api.cpp', 'picpac.cpp', 'picpac-cv.cpp', 'json11/json11.cpp'],
        depends = ['json11/json11.hpp', 'picpac.h', 'picpac-cv.h'])

setup (name = 'picpac_ssd',
       version = '0.2.2',
       url = 'https://github.com/aaalgo/picpac',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'BSD',
       description = 'This is a demo package',
       ext_modules = [picpac],
       requires = ["cv2"],
       )
