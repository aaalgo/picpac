#!/bin/bash

if [ ! -d json11 ]
then
    git submodule init
    git submodule update
fi
docker run -it -v $PWD:/picpac aaalgo/cbox:latest /picpac/docker-make-server.sh
