# Introduction

PicPac is an image streamer that feeds data to various deep learning frameworks
for iterative training.  It tries to solve the following two problems: 

1. There lacks a unified image streamer for deep learning frameworks.
Existing frameworks typically use
generic storage backends like HDF5, leveldb/lmdb, etc.  They either provide a
thin adaptive layer for streaming, or rely on third party generic streaming
library like [fuel](https://github.com/mila-udem/fuel).  Typically, changing a
learning framework requires working out the data streaming mechanism afresh,
and it is a challenge to make different frameworks see the same dataset the same
way so as to fairly compare the performance of the down-stream processing.

2. It is a burden to manage different versions of the same image dataset for
experimental purposes.  When a framework does not support on-the-fly resizing,
data splitting for cross validation or other preprocessing,
one has to temporarily store the preprocessing results to the filesystem before
feeding to the framework for learning.  One will soon face the delimma of whether
to spend time waiting for preprocessing, or two manage the ever growing versions
of the same dataset.  We observe that most CPU cores are free when training happens
on GPU, so we make the design desicion to always store the raw data when possible,
and do all kinds of preprocessing on-the-fly in parallel with the main training
process.

We make our design decisions in favor of flexibility, and aim at small to medium
datasets.  <font color='red'>We assume SSD storage, or memory that is big enough to
hold the whole dataset</font>.  To stream extremely large datasets like the ImageNet
with the sequential reading throughput of HDDs, see [RecordIO](http://myungjun-youn-demo.readthedocs.org/en/latest/python/io.html#create-dataset-using-recordio) or [PicPoc](https://github.com/aaalgo/picpoc).

# QuickStart

A PicPac database is a single file. Below is how to import data for a classification
problem.

## Importing Data
```
picpac-import -f 2 data_dir path_to_db
```

The input format 2 assumes that data_dir has N subdirectories named 0, 1, ..., each
containing training images for one category.  See [Importing Data](import.md) for
other input formats.

## Streaming

```python
import picpac

config = dict(loop=True,   # restart from beginning when all data consumed
                           # this leads to an endless loop.
         batch=16,
         # with batch > 0, images in the same batch must be of the same size
         # use resize_width/height if raw images have different sizes
         resize_width=256,
         resize_width=256,
	 )

stream = picpac.ImageStream('path_to_db', **config)

for images, labels, _ in stream:
    # do something with images and labels
    # images is 4-dim tensors in the shape of (batch, channel, rows, cols)
    # labels is 1-dim array by default
    pass
```

The streaming behavior can be tweaked with the config dict.
See [Configuration](config.md) for all supported configuration parameters.

