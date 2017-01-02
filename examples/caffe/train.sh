#!/usr/bin/env sh

export GLOG_log_dir=log
export GLOG_logtostderr=1

mkdir -p snapshots

SNAP=$1
if [ -z "$SNAP" ]
then
    nice caffe train --solver solver.prototxt $* | tee log
else
    shift
    nice caffe train -solver solver.prototxt -snapshot $SNAP $* | tee log
fi

