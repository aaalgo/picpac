from distutils.core import setup, Extension

picpac = Extension('picpac',
        extra_compile_args = ['-O3', '-std=c++1y'], 
        define_macros = [('MAJOR_VERSION', '1'),
                         ('MINOR_VERSION', '0')],
        include_dirs = ['/usr/local/include'],
        libraries = ['opencv_highgui', 'opencv_core', 'boost_filesystem', 'boost_system', 'boost_python', 'glog'],
        library_dirs = ['/usr/local/lib', 'json11'],
        sources = ['python-api.cpp', 'picpac.cpp', 'picpac-cv.cpp', 'json11/json11.cpp'])

setup (name = 'PicPac',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [picpac],
       py_modules = ['picpac_mxnet'])
