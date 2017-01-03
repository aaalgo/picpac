.PHONY:	help docker run

help:
	@echo "Usage:"
	@echo "  make docker # build docker"
	@echo "  make run    # run docker with current directory mounted at /picpac"

docker:
	nvidia-docker build -t aaalgo/mxnet-picpac .

run:
	nvidia-docker run -it -v $(PWD):/root/picpac aaalgo/mxnet-picpac /bin/bash
