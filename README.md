PicPac: An Image Database and Streamer for Deep Learning
========================================================

PicPac is an image database and streamer for deep learning.
It is developed so that the user of different deep learning frameworks
can all use the same image database format. 

# Installation 

## Option 1: download binary python module.

This is the recommended installation method if you are using ubuntu
16.04 and python3.5.

Download the .so file from
[here](http://www.aaalgo.com/picpac/binary/picpac.cpython-35m-x86_64-linux-gnu.so)
and drop in your current directory.  You should be able to `import picpac` in python3.

## Option 2: building from source code.

Prerequisits:
- boost libraries  (libboost-all-dev on ubuntu or boost-devel on centos )
- opencv2  (libopencv-dev or opencv-devel)
- glog  (libglog-dev or glog-devel)

```
git clone --recurse-submodules https://github.com/aaalgo/picpac
cd picpac

# python 2, not recommended
python setup.py build
sudo python setup.py install

# python 3
python3 setup.py build
sudo python3 setup.py install
```

## Option 3: pip

We are working on it.


# Quick Start

## Basic Structure

A PicPac database is a collection of image samples.

- Image itself.
- A float32 label.


## Data Importing



# [Legacy Documentation](http://picpac.readthedocs.org/en/latest/)


# Examples

- [Tensorflow Slim](https://github.com/aaalgo/cls)

# Public Dataset

- [Public dataset imported to PicPac databases](http://www.aaalgo.com/picpac/datasets/)
- [How to import data](http://picpac.readthedocs.io/en/latest/import/)


# PicPac Explorer

PicPac Explorer is a Web-based UI that allows the user to explore the
picpac database content and simulate streaming configurations.

Download portable distribution of PicPac Explorer here: (http://aaalgo.com/picpac/binary/).

Run ```picpac-explorer db``` and point the web browser to port 18888.  If the program is executed under a GUI environment, the browser will be automatically opened.

# Building C++ Binaries

The basic library depends on OpenCV and Boost.  The dependency on [Json11](https://github.com/dropbox/json11)
is provided as git submodule, which can be pulled in by 
```
git submodule init
git submodule update
```

PicPac Explorer for visualizing annotation results is built with separate rules and has many more
dependencies.  Use the link about to download a portable pre-built version.

