#!/bin/bash

docker run -it -v $PWD:/picpac aaalgo/cbox:latest /picpac/make-server-helper.sh
