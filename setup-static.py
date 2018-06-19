import sys
import os
import subprocess as sp
import numpy
from distutils.core import setup, Extension

git_version = sp.check_output('git rev-parse HEAD', shell=True)
git_changed = sp.check_output('git diff | wc -l', shell=True)

if git_changed != '0':
    print("Git has changed.  Please build first!")
    sys.exit(0)

numpy_include = os.path.join(os.path.abspath(os.path.dirname(numpy.__file__)), 'core', 'include')

libraries = ['opencv_imgcodecs', 'opencv_imgproc', 'opencv_core', 'libjasper', 'libjpeg', 'libpng', 'libtiff', 'libwebp', 'zlib', 'boost_filesystem', 'boost_system', 'boost_python35', 'glog']

picpac = Extension('picpac',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y', '-I/opt/cbox/include'], 
        include_dirs = ['/usr/local/include', 'pyboostcvconverter/include', 'json11', numpy_include],
        libraries = libraries,
        library_dirs = ['/usr/local/lib', '/opt/cbox/lib'],
        sources = ['python-api.cpp', 'picpac.cpp', 'picpac-image.cpp', 'shapes.cpp', 'transforms.cpp', 'picpac-cv.cpp', 'json11/json11.cpp'],
        depends = ['json11/json11.hpp', 'picpac.h', 'picpac-image.h', 'bachelor/bachelor.h'])

setup (name = 'picpac',
       version = '0.2.2',
       url = 'https://github.com/aaalgo/picpac',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'BSD',
       description = 'This is a demo package',
       ext_modules = [picpac],
       )

