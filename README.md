

Design Decisions
================
- Store raw data, do transformation and augmentation on-the-fly.
This is based on the observation that CPU cores are mostly free when trained on GPU.
- Use random access and memory cache, assuming big memory and/or SSD storage.
For sequential I/O with extremely large dataset, like the whole ImageNet, see [RecordIO](http://myungjun-youn-demo.readthedocs.org/en/latest/python/io.html#create-dataset-using-recordio) or [PicPoc](https://github.com/aaalgo/picpoc).

Building
========

The basic library depends on OpenCV 2.x and Boost.  The dependency on [Json11](https://github.com/dropbox/json11)
is provided as git submodule, which can be pulled in by 
```
git submodule init
git submodule update
```

There's a web server for exploring the database and visualize augmented image/annotations.
The web server depends on [libmagic](https://github.com/threatstack/libmagic) and [served](https://github.com/datasift/served) to build.  The server is not needed for training purpose, but could be useful to make sure the imported
data is correct.

Data Importing
==============
```
$ picpac-import -h
Allowed options:
  -h [ --help ]                  produce help message.
  -i [ --input ] arg
  -o [ --output ] arg
  --max arg (=-1)
  --resize arg (=-1)
  -f [ --format ] arg (=1)
  --cache arg (=".picpac_cache")

Formats:
  0: scan a directory
  1: list of <image	label>
  2: scan a directory
  3: list of <image	json-annotation>
  4: list of <image	annotation-image>
```

We currently support 4 input formats:

0. Recursively load all images under directory "input", assigning label 0 to everything.
1. From a text file, each line specifying an image path/url and a float label, separated by tab.
2. Scan directory "input" for sub-directories named by category IDs 0, 1, ....  Recursively load
each sub-directory and use the category ID as image label.
3. From a text file, each line with an image path/url and a json annotation separated by tab.  Assigning label 0 to everything.
4. From a text file, each line with an image path/url and a label image path/url, seprated by tab.

When images are given by URL, wget is used to download the image.  The downloaded content is cached
in the specified directory, so when the same URL is imported again the content can be directly loaded from cache.

Although a resize options is provided, it is not recommended.  Use config.resize_width/height for
on-the-fly resizing when streaming.  For images that are really too large, the max option can be used to
limit the image size, so on-the-fly resizing can be more efficient.  The rationale is that the database
does not have to be recreated when training has to be done at different scales.

There is also a Python API for data importing, so the output of cv2.imencode can be directly
added to the database without going through the filesystem.

Json Annotation
===============

We support [Annotorious](http://annotorious.github.io/) styple JSON annotation.
Examples are
```
{"shapes":[{"type":"rect","geometry":{"height":0.1329243353783231,"width":0.34801136363636365,"y":0.5766871165644172,"x":0.375}}]}

{"shapes":[{"type":"ellipse","geometry":{"height":0.1329243353783231,"width":0.34801136363636365,"y":0.5766871165644172,"x":0.375}}]}  // ellipse within a bounding rect

{"shapes":[{"type":"polygon","geometry":{"points":[{"y":0.5010660980810234,"x":0.7230113636363636},{"y":0.5010660980810234,"x":0.7230113636363636},{"y":0.4925373134328358,"x":0.796875},{"y":0.5714285714285714,"x":0.7926136363636364},{"y":0.5884861407249466,"x":0.7116477272727273},{"y":0.5010660980810234,"x":0.7159090909090909},{"y":0.4989339019189765,"x":0.7144886363636364},{"y":0.5031982942430704,"x":0.7130681818181818},{"y":0.4989339019189765,"x":0.7130681818181818}]}}]}

```

- Note that all values are relative to width/height, so the same annotation can be applied to resized images. 
- One annotation can contain multiple shapes of same or different types.
- When location/size values are relative to width/height, we cannot easily specify a radius for a circle, so we
choose to support ellipses instead (for now).
- All other fields returned by Annotorious are ignored.

To enable json-style annotation, set config.annotate = "json".  The annotation will then be rendered on-the-fly.


Image Annotation
================

We also support image-based annotation, which can be enabled by setting config.annotate = "image".
An training example and it's annotation image must be of the same size.  The annotation image
can only have a single channel.  When annotation pixels are to be interpreted as class IDs,
image must be stored in a lossless format like PNG.



Basic Streaming Usage
=====================

```
import picpac

seed = 1996
config = dict(seed=seed,
	loop=False,
	shuffle=True,
	reshuffle=True,
	batch=batch,
	# unless batch is 1, always set resize_width and resize_height
	# as samples in the same batch must have the same size
	resize_width=256,
	resize_height=256,
	split=K,
	# for K-fold cross validation, run K times with split_fold = 0, 1, ..., K-1
	split_fold=0,
	channels=1,
	stratify=False,
	pert_color1=10,
	pert_angle=5,
	pert_min_scale=0.8,
	pert_max_scale=1.2,
	pad=False,
	pert_hflip=True,
	)
train_stream = picpac.ImageStream(db_path, negate=False, perturb=True, **config)
valid_stream = picpac.ImageStream(db_path, negate=True, perturb=False, **config)

while True:
    try:
        images, labels, pad = train_stream.next()
        # when config.pad is True, partial batches might be returned.
        # the returned images and labels are still the batch size,
        # and the number of padding items are returned as "pad".
	# "pad" is always 0 if config.pad is False.

        # do training
    except StopIteration:
        break

```

For each batch, images is a 4-dimensional tensor of the shape (batch_size, channels, rows, cols).
Depending on configuration, labels can be of one of the following four sizes:
- (batch_size): neither "annotate" or "onehot" is set.  The raw float label returned.
- (batch_size, classes): "annotate" is not set, "onehot" is set to number of classes. The raw float label is interpreted as class ID, and therefore must be integral.
- (batch_size, 1, rows, cols): "annotate" is set and "onehot" is not set; for pixel-level regression.
- (batch_size, classes, rows, cols): both "annotate" and "onehot" are set; for pixel-level classification.  One-hot
encoding is done assuming pixel value is the integral class ID.


MXNet Usage
===========

```
from picpac.mxnet import ImageStream


# create ImageStream using the same configuration parameters
# picpac.mxnet.ImageStream is inherited from mxnet.io.DataIter

```

Neon Usage
==========

```
from picpac.neon import ImageStream

# create ImageStream using the same configuration parameters
# picpac.meon.ImageStream is inherited from neon.data.dataiterator.NervanaDataIterator

```

Caffe Usage
===========

Use this Caffe (fork)[https://github.com/aaalgo/caffe-picpac].

```
layer {
  name: "data"
  type: "PicPac"
  top: "data"
  top: "label"
  picpac_param {
    path: "path_to_db"
    batch: 16
    channels: 3
    split: 5
    split_fold: 0
    resize_width: 224
    resize_height: 224
    threads: 4
    perturb: true
    pert_color1: 10
    pert_color2: 10
    pert_color3: 10
    pert_angle: 20
    pert_min_scale: 0.8
    pert_max_scale: 1.2
  }
}

```

Note that the above specification always go through the following touchup:
```
  config_.loop = true;
  if (this->phase_ == TRAIN) {
      config_.reshuffle = true;
      config_.shuffle = true;
      config_.stratify = true;
      config_.split_negate = false;
  }
  else {
      config_.reshuffle = false;
      config_.shuffle = true;
      config_.stratify = true;
      config_.split_negate = true;
      config_.perturb = false;
      config_.mixin.clear();
  }

```



Parameter Documentation
=======================
The following parameters are exposed to Python and Caffe.

* General Streaming Parameters
	- int32 seed: randomization seed
	- bool loop: loop through records for endless streaming, instead of throwing StopIteration when all data are consumed.
	- bool shuffle: shuffle input records order before splitting the data.
	- bool reshuffle: shuffle input records in the beginning of every loop from the second loop on.
	- bool stratify: divide data by category, split and loop within each category, so any sliding window on the stream contain about same number of images for each category.
	- bool cache: cache all data in memory after they are read.  Default is true, set to false when data do not fit in memory.
	- uint32 preload: pre-load queue size.
	- uint32 threads: number of decoding/pre-processing threads, default = 4.
	- uint32 batch: batch size
	- bool pad: if loop is false and there's not enough to produce a whole batch, throw StopIteration if not pad, otherwise return a whole batch with random padding.  Number of padded samples are also returned.

* Cross validation
	- uint32 split: equally split all data (or each category if stratify = true) into this number of partitions.
	- int32 split_fold: exclude examples from this split.
	- bool split_negate: use the rest of samples.

For example, 5-fold cross validation can be run with split = 5, and with split_fold = 0, 1, 2, 3, 4.
For each value of split_fold, set split_netage = false when training, and true when validating.

* Image Control
	- int32 channels: number of image channels, ignored if <= 0
	- int32 min_size: 
	- int32 max_size:
	- int32 resize_width: resize to this, ignored if <= 0
	- int32 resize_height: resize to this, ignore if <= 0
	- int32 decode_mode: as in the second parameter of cv::imread.  Set to -1 to load images as is.
	- bool bgr2rgb: default 3-channel images are stored as BGR (opencv behavior), set to true for RGB.

* Image Annotation
	- string annotate: for annotation, set to "json" or "image"
	- int32 anno_type: annotation image type, as in cv::Mat::type()
	- bool anno_copy: copy image to label, for visualization.  Set to false for training.
	- float anno_color1: annotation channel 1.
	- float anno_color2: annotation channel 2, typically not used. 
	- float anno_color3: annotation channel 3, typically not used.
	- int32 anno_thickness: set to -1 to fill the annotation region with anno_color.

* Image Augmentation
	- bool perturb: set to true to enable augmentation
	- float pert_color1: 
	- float pert_color2
	- float pert_color3
	- float pert_angle: image/label rotated randomly within [-pert_angle, pert_angle], in degrees.
	- float pert_min_scale: image/label rescaled to a scale uniformly sampled from [pert_min_scale, pert_max_scale].
	- float pert_max_scale
	- bool pert_hflip: enable random horizontal flip.
	- bool pert_vflip: enable random vertical flip.

* Label Encoding
	- uint32 onehot: one hot encoding of label.  If onehot > 0, specify the number of categories.


