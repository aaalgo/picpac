PicPac: An Image Streamer for Iterative Training

PicPac is an image database, explorer and streamer for deep learning.
It is developed so that the user of different deep learning frameworks
can all use the same image database format.  PicPac currently supports
streaming data into TensorFlow, MXNet, Torch and a Caffe fork, with C++,
python and Lua API. 

#[Documentation](http://picpac.readthedocs.org/en/latest/)

# Installation with Pip
Prerequisits:
- boost libraries  (libboost-all-dev on ubuntu or boost-devel on centos )
- opencv2  (libopencv-dev or opencv-devel)
- glog  (libglog-dev or glog-devel)

```
pip install -i https://testpypi.python.org/pypi picpac
```

This will install the Python streaming API.

# Examples

- [Tensorflow Slim](https://github.com/aaalgo/picpac/tree/master/examples/tensorflow)
- [MXNet](https://github.com/aaalgo/picpac/tree/master/examples/mxnet)

# Public Dataset

- [Public dataset imported to PicPac databases](http://www.aaalgo.com/picpac/datasets/)
- [How to import data](http://picpac.readthedocs.io/en/latest/import/)


# PicPac Explorer

PicPac Explorer is a Web-based UI that allows the user to explore the
picpac database content and simulate streaming configurations.

Download portable distribution of PicPac Explorer here: (http://aaalgo.com/picpac/binary/).

Run ```picpac-explorer db``` and point the web browser to port 18888.  If the program is executed under a GUI environment, the browser will be automatically opened.

# Building

The basic library depends on OpenCV 2.x and Boost.  The dependency on [Json11](https://github.com/dropbox/json11)
is provided as git submodule, which can be pulled in by 
```
git submodule init
git submodule update
```

PicPac Explorer for visualizing annotation results is built with separate rules and has many more
dependencies.  Use the link about to download a portable pre-built version.

