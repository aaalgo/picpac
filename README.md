PicPac: An Image Streamer for Iterative Training

#[Documentation](http://picpac.readthedocs.org/en/latest/)

# Installation with Pip
Prerequisits:
- boost libraries  (libboost-all-dev on ubuntu or boost-devel on centos )
- opencv2  (libopencv-dev or opencv-devel)
- glog  (libglog-dev or glog-devel)

```
pip install -i https://testpypi.python.org/pypi picpac
```

#PicPac Server

Download portable distribution of PicPac server here: (http://aaalgo.com/picpac/server/).

#Building

The basic library depends on OpenCV 2.x and Boost.  The dependency on [Json11](https://github.com/dropbox/json11)
is provided as git submodule, which can be pulled in by 
```
git submodule init
git submodule update
```

PicPac server for visualizing annotation results is built with separate rules and has many more
dependencies.  Use the link about to download a portable pre-built version.
