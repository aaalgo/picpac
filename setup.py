from distutils.core import setup, Extension

picpac = Extension('_picpac',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y'], 
        include_dirs = ['/usr/local/include', 'json11'],
        libraries = ['opencv_highgui', 'opencv_core', 'boost_filesystem', 'boost_system', 'boost_python', 'glog'],
        library_dirs = ['/usr/local/lib'],
        sources = ['python-api.cpp', 'picpac.cpp', 'picpac-cv.cpp', 'json11/json11.cpp'],
        depends = ['picpac.h', 'picpac-cv.h', 'json11/json11.hpp'])

setup (name = 'picpac',
       version = '0.2',
       url = 'https://github.com/aaalgo/picpac',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'BSD',
       description = 'This is a demo package',
       ext_modules = [picpac],
       py_modules = ['picpac.mxnet', 'picpac.neon'],
       requires = ["cv2"],
       )
