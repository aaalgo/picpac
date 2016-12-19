#!/bin/bash

docker run -it -v $PWD:/picpac aaalgo/cbox:latest /picpac/docker-make-server.sh
