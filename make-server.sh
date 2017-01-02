#!/bin/bash

if [ ! -f json11/json11.cpp ]
then
    git submodule init
    git submodule update
fi
docker run -it -v $PWD:/picpac aaalgo/cbox:latest /picpac/docker-make-server.sh
