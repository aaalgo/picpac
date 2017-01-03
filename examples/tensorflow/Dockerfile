FROM tensorflow/tensorflow:0.12.1-devel-gpu
LABEL maintainer "Wei Dong <wdong@wdong.org>"
RUN apt-get update && apt-get install -y libboost-all-dev python-opencv libopencv-dev libgoogle-glog-dev libgflags-dev
RUN pip install -i https://testpypi.python.org/pypi picpac
WORKDIR /root/picpac
