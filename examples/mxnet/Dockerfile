FROM kaixhin/cuda-mxnet:8.0
LABEL maintainer "Wei Dong <wdong@wdong.org>"
RUN apt-get update && apt-get install -y libboost-all-dev python-opencv libgoogle-glog-dev libgflags-dev
RUN apt-get install -y python-pip
RUN pip install -i https://testpypi.python.org/pypi picpac
WORKDIR /root/picpac
