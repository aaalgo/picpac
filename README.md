PicPac: An Image Database for Deep Learning
===========================================

PicPac is an image database for deep learning.  It is developed so that
the user of different deep learning frameworks can all use the same
image database format. 

# Installation 

## Option 1: download binary python module.

This is the recommended installation method if you are using ubuntu
16.04 and python3.5.

Download the .so file from
[here](http://www.aaalgo.com/picpac/binary/picpac.cpython-35m-x86_64-linux-gnu.so)
and drop in your current directory.  You should be able to `import picpac` in python3.

```
wget http://www.aaalgo.com/picpac/binary/picpac.cpython-35m-x86_64-linux-gnu.so
echo "import picpac; print(picpac.__file__)" | python3
```

## Option 2: building from source code.

Prerequisits:
- boost libraries  (libboost-all-dev on ubuntu or boost-devel on centos )
- opencv2  (libopencv-dev or opencv-devel)
- glog  (libglog-dev or glog-devel)

```
git clone --recurse-submodules https://github.com/aaalgo/picpac
cd picpac

# python 2, not recommended
python setup.py build
sudo python setup.py install

# python 3
python3 setup.py build
sudo python3 setup.py install
```

# Quick Start

## Basic Concepts

A PicPac database is a collection of records.
A record is the unit of saving/loading/streaming operation,
and contains data of a single training sample, which is
typicall an image with a label and/or a set of annotations.
A record contains the following:

- `id`: serial number of uint32, automatically set to 0, 1, ... when imported in
  python.
- `label`: a label of float32.  We use float to support both
  classification and regression.
- ~~`label2`: a secondary label of type int16.  Typically not used.~~
- `fields[]`: up to 6 binary buffers.
- `fields[0]`: this is typically the binary image, supports most formats.
- `fields[1]`(optional): annotation in JSON or binary image.
- `fields[2-5]`: typically not used.

We recommend storing raw images in the database, unless the image is very big
in size.  PicPac does all decoding, augmentation and other transformations on-the-fly.

## Data Importing

```python
import picpac

db = picpac.Writer('path_to.db', picpac.OVERWRITE)

for label, image_path, mask in some_list:
    with open(image_path, 'rb') as f:
        image_buf = f.read()

    if mask is None:
        # import without annotation, for classification tasks.
        db.append(float(label), image_buf)
        continue

    # there's annotation/mask
    if mask_is_a_zero_one_png_image:
        with open(mask, 'rb') as f:    # use lossless PNG for annotation
            mask_buf = f.read()
        db.append(float(label), image_buf, mask_buf)
        continue

    if mask_is_json_annotation:
        import simplejson as json
        db.append(float(label), image_buf, json.dumps(mask).encode('ascii'))
        continue

    # or if you want more fields, python supports up to 4 buffers
    # use case: several consecutive video frames as a single example
    # they'll go through identical augmentation process.

    db.append(float(label), image_buf, extra_buf1, extra_buf2, extra_buf3)
    pass

```

You can view database content with picpac-explorer; see below.

## Streaming for classification
After a database has been created, it can be used to stream
training samples to a deep-learning framework:

```python
import picpac

is_training = True

config = {"db": db_path,
          "loop": is_training,          # endless streaming
          "shuffle": is_training,       # shuffle upon loading db
          "reshuffle": is_training,     # shuffle after each epoch
          "annotate": False,
          "channels": 3,                # 1 or 3
          "stratify": is_training,      # stratified sample by label
          "dtype": "float32",           # dtype of returned numpy arrays
          "batch": 64,                  # batch size
          "cache": True,                # cache to avoid future disk read
          "transforms": [ 
              {"type": "augment.flip", "horizontal": True, "vertical": False, "transpose": False},
              # other augmentations
              {"type": "resize", "size": 224},
              ]
         }

stream = picpac.ImageStream(config)

for meta, images in stream:  # the loop is endless

    # meta.labels is the image labels of shape (batch, )
    # images is of shape (batch, H, W, channels)

    # feed to tensorflow
    feed_dict = {X: images, Y: meta.labels, is_training: True}
    sess.run(train_op, feed_dict=feed_dict)

    if need_to_stop:
        break 
```

PicPac doesn't do automatic image resizing.  Usually images in the
database are of different shapes.  But all images in the minibatch
must be of the same shape. So you have two options:

- Use batch size of 1.
- Add a `resize` transform like the example above.

## Streaming for segmentation

Annotation is enabled by adding `annotate: [1]` in the configuration.
1 here is the field ID that contains annotation.
Both image and annotation will go through identical augmentation
process.  When image interpolation is needed, image pixel values
are produced with linear interpolation while label pixels are
produced with nearest-neighbor interpolation(so we don't accidentally
produce meaningless categorical labels like 0.5).


```python
import picpac

is_training = True

config = {"db": db_path,
          # ... same as above ...
          #batch": 1,         # so we don't need to resize image
                              # and resize/clip transform for batch > 1
          "annotate": [1],    # load field 1 as annotation
          "transforms": [ 
              {"type": "augment.flip", "horizontal": True, "vertical": False, "transpose": False},
              {"type": "clip", "round": 16},	     # feature stride, see below
              # {"type": "resize", "size": 224},     # add this for batch > 1
              {"type": "rasterize"}
              ]
         }


stream = picpac.ImageStream(config)

for _, images, labels in stream:
    # images is of shape (batch, H, W, channels)
    # labels is of shape (batch, H, W, 1)

    # feed to tensorflow
    feed_dict = {X: images, Y: labels, is_training: True}
    sess.run(train_op, feed_dict=feed_dict)
```

We typically use vector graphics (encoded in JSON) for annotation.
All augmentations and other transformations are directly applied to vector
graphics, and the final `rasterize` step converts the vector graphics
(when applicable) to dense image.  `rasterize` will have no effect if
the annotation is dense image.

Typicall a segmentation model goes through a serious of convolution and
deconvolution, and one will want the produced label image to be well aligned
with the input.
Make use you set `round` parameter of `clip`
transformation to your feature stride, so the generated minibatch will
have width and height clipped to be divisible by this stride value.

## Streaming for Bounding Box Regression

The database must be created with JSON-based annotation.  Image-based
annotation is not supported.  It is recommended that you convert image
masks into contours and encode them as polygons.

See https://github.com/aaalgo/box/blob/master/train-anchors.py for a full
example.

We are still working on an API that supports multiple priors.

```python
import picpac

config = {"db": db_path,
          # ... same as above ...
          # batch": 1,        # so we don't need to resize image
                              # and resize/clip transform for batch > 1
          "annotate": [1],    # same as segmentation
          "transforms": [ 
              # augmentations ...
              {"type": "clip", "round": 16},    # feature stride, see below
              {"type": "anchors.dense.box", 'downsize': anchor_stride},
              {"type": "rasterize"}
              ]
         }


stream = picpac.ImageStream(config)

for _, images, labels, anchors, anchor_weight, params, params_weight in stream:
    # images is of shape (,,,channels)
    # labels is generated by rasterize, same as segmentation
    # anchors is of shape (,,,priors), 0/1 anchor mask, priors = 1
    # anchors_weight is of shape (,,,priors)
    # params is of shape (,,,priors * 4),  box parameters
    #                for each prior, the 4 numbers are (dx, dy, width, height)
    # params_weight is of shape (,,,priors)
```

Note that we still need to rasterize any JSON-based annotation thats
loaded even
though we do not need them here; PicPac is not able to encode
JSON strings into a minibatch.  In the future we might be able to
replace this with a `drop` operation and save computation.

`anchor_weight` and `params_weight` are masks to decide which
pixel-prior combination should participate in loss calculation for
anchors and params.

## Streaming for Mask-RCNN

This is one extra step on top of box regression (`box_feature`
transformation and setting `use_tag` of `rasterize`).

See https://github.com/aaalgo/box/blob/master/train.py for a full
example.

```python
import picpac

is_training = True

config = {
          # ... same as box regression ...
          "transforms": [ 
              # augmentations ...
              {"type": "clip", "round": 16},    # feature stride, see below
              {"type": "anchors.dense.box", 'downsize': anchor_stride},
              {"type": "box_feature"},
              {"type": "rasterize", "use_tag": True, "dtype": "float32"}
              ]
         }


stream = picpac.ImageStream(config)

for _, images, tags, anchors, anchor_weight, params, params_weight, box_feature in stream:
    # images is of shape (,,,channels)
    # tags is of shape (,,,1)
    # anchors is of shape (,,,priors), 0/1 anchor mask, priors = 1
    # anchors_weight is of shape (,,,priors)
    # params is of shape (,,,priors * 4),  box parameters
    # params_weight is of shape (,,,priors)
    #

    # box_feature is of shape (N, 7), where N is the number of boxes
    #    box_feature[:, 0]      image index within minibatch, 0-based
    #    box_feature[:, 1]      object label
    #    box_feature[:, 2]      object tag
    #    box_feature[:, 3:5]    (x1, y1), top left coordinate, clipped to image area
    #    box_feature[:, 5:7]    (x2, y2), bottom right coordinate, clipped to image area

	# params are not clipped to image area.
	# box_feature[:, 3:7] are clipped, otherwise the two are the same.
```

We use the following label-tag mechanism to achieve efficient extraction
of mask patches with augmentation:

- Each annotated shape (usually a polygon) has a label and a tag.
- Label is the categorical label; the prediction target.
- Tag an non-zero integral value calculated when importing samples so as
  to differenciate pixels of touching objects.
  [Four color theorem](https://en.wikipedia.org/wiki/Four_color_theorem) states that four different tag values (in addition to the background 0) are sufficient if we have to tag touching objects differently.  In our case, in order to achieve good separation between objects, we want to assign different tags to two objects if they touch after [dilation](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html).  The number of tag values we use do not affect computational cost. We can assume the range [1, 255] is always available.  [This program](https://github.com/aaalgo/box/blob/master/gcolor.py) does such tagging/coloring.
- Instead of a label image, `rasterize` here generates a tag image
  (`use_tag: True`).   The label information is returned in
  `box_feature[:, 1]`. 

PicPac does not directly return the masks, but the masks can be easily
produced with the following procedure:

1. `box_feature[i, 3:7]` is the bounding box information, already
   clipped to image area.  That is `0 <= box_feature[i, 3] <=
   box_feature[i, 5] < width`.  Round the number and get the
   corresponding ROI in the tag image.
2. Set all pixels to 1 where the tag is `box_feature[i, 2]` and
   the remaining pixels to 0.
3. Resize all masks to the same size.

These are implemented by the `MaskExtractor` class in
https://github.com/aaalgo/box/blob/master/python-api.cpp.  The same
program implements some other routines that are needed for Mask-RCNN
implementation.

The importing program has to guarantee that within the bounding box
of an object there's no part of another object with the same tag.
This can usually be achieved by setting a sufficiently large dilation
value when testing the touching condition.

## Reading Database 

Use `picpac.Reader` to access raw data.

```python
import picpac

db = picpac.Reader(path)

# method 1
for rec in db:
    rec.label	    # is the label
	rec.fields      # are the fields
	rec.fields[0]   # is usually the image

# method 2
for i in range(db.size()):
    rec = db.read(i)
```

# Special Topics

## Annotation

PicPac supports image-based annotation.  But to achieve better
flexibility, we prefer JSON-based annotation.  PicPac's annotation
format is based on that of
[annotorious](https://annotorious.github.io/).
[OWL](https://github.com/aaalgo/owl/) is our in-house tool to produce
such annotations. 

Below is a sample json annotation with a rectangle and a polygon.
```javascript
{"shapes": [ {"label": 1.0,
              "type": "rect",
              "geometry": {"x": 0.15, "y": 0.13, "height": 0.083, "width": 0.061},
             },
             {"label": 1.0,
              "type": "polygon",
              "geometry": {"points": [{"y": 0.75, "x":0.62},
                                      {"y": 0.75, "x":0.61},
                                      ....
                                      {"y": 0.75,"x": 0.61}
                                     ]}

             },
             ...
            ]
}

```

Note that all x, y, width and height values are normalized to a [0, 1]
range, with x and width divided by image width and y and hight divided
by image height.

In addition to `label`, each shape might also carry an optional integral
`tag` value, which can be optionally rendered by the `rasterize`
operation.

PicPac ignores any additional data in JSON that it does not recognize.

## Facets and Transformation

PicPac database stores raw data of training samples in the in up to 6
buffers on disk.  Usually only the first two are used for the image and
the annotation, but our API is flexible enough to support multiple
images and annotations.  The only constraints now is that images of
the same sample must have the same shape.

At streaming time, PicPac use a series of loading and transformation
operations to create a set of Facets for each example.  These facets are
merged into minibatches and returned in the streaming API.
The general streaming API is

```python
for meta, facet0, facet1, ..., last_facet in stream:
    # meta is the meta data
    pass
```

The facets loading are controled by the following two fields in
configuration.
```
    config = { ...
               'images': [0],    # default to [0]
               'annotate': [1]   # default to []
               ...
			   'transforms': [ ...]
             }
```

Both `images` and `annotate` are lists of field IDs (so they must be
within 0-5).

First, fields in `images` are loaded into the facets list,
and then fields in `annotate`.   After that, transformations are
applied to the facets, some, like `anchor.dense.box` generating new facets.

## Augmentation

A subset of the supported transformations implement image augmentation.
An augmentation operation applies to all facets in the same way whenever
applicable.

## Accessing Raw Data in Streaming



## I/O Performance and Caching

PicPac enables caching by default, which means images are loaded from
disk only once.  But with big dataset, this can cause an out-of-memory
error.  In such case, one has to set `cache = False` in configuration,
and make sure the database file is on SSD-storage.  PicPac loads each
sample with a random seek.

## Mixin


## Label2 and Stratified Sampling

This is not yet exposed to the Python API.

In order to support stratified sampling, a PicPac database contains an
index with the sample category information.  The object category must
be decided at database create time and is usually determined by the
`label` field.  In rare cases, the sampling category can be different
from labels.  In such case, a database is created with the
`INDEX_LABEL2` flag, and the `label2` field of the record is set to the
stratified sampling category.



# [Legacy Documentation](http://picpac.readthedocs.org/en/latest/)


# Examples

- [Tensorflow Slim](https://github.com/aaalgo/cls)


# PicPac Explorer

PicPac Explorer is a Web-based UI that allows the user to explore the
picpac database content and simulate streaming configurations.

Download portable distribution of PicPac Explorer here: (http://aaalgo.com/picpac/binary/picpac-explorer).

Run ```picpac-explorer db``` and point the web browser to port 18888.  If the program is executed under a GUI environment, the browser will be automatically opened.

# Building C++ Binaries

The basic library depends on OpenCV and Boost.  The dependency on [Json11](https://github.com/dropbox/json11)
is provided as git submodule, which can be pulled in by 
```
git submodule init
git submodule update
```

PicPac Explorer for visualizing annotation results is built with separate rules and has many more
dependencies.  Use the link about to download a portable pre-built version.

