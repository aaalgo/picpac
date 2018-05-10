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
        with open(mask, 'rb') as f:		# use lossless PNG for annotation
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

```
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
1 here is the field ID that contains annotation, which is almost always
the case.
Both image and annotation will go through identical augmentation
process.  When image interpolation is needed, image pixel values
are produced with linear interpolation while label pixels are
produced with nearest-neighbor interpolation(so we don't accidentally
produce meaningless categorical labels like 0.5).

We are still working on an API that supports multiple priors.

```
import picpac

is_training = True

config = {"db": db_path,
          # ... same as above ...
          #batch": 1,         # so we don't need to resize image
                              # and resize/clip transform for batch > 1
          "annotate": [1],    # load field 1 as annotation
          "transforms": [ 
              {"type": "augment.flip", "horizontal": True, "vertical": False, "transpose": False},
              {"type": "clip", "round": 16},	# feature stride, see below
              # {"type": "resize", "size": 224},  -- add this for batch > 1
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

We typically use vector graphics (JSON-based) for annotation.
All augmentations and transformations are directly applied to vector
graphics, and the final `rasterize` step converts the vector graphics
(when applicable) to dense image.  `rasterize` will have no effect if
the annotation is dense image.

Typicall segmentation model go through a serious of convolution and
deconvolution, and one will want the produced label image to be well aligned
with the input.
Make use you set `round` parameter of `clip`
transformation to your feature stride, so the generated minibatch will
have width and height clipped to be divible by the stride value.

## Streaming for Bounding Box Regression

The database must be created with JSON-based annotation.  Image-based
annotation is not supported.  It is recommended that you convert image
masks into contours and encode them in JSON.

See https://github.com/aaalgo/box/blob/master/train-anchors.py for a full
example.

```
import picpac

is_training = True

config = {"db": db_path,
          # ... same as above ...
          #batch": 1,         # so we don't need to resize image
                              # and resize/clip transform for batch > 1
          "annotate": [1],    # same as segmentation
          "transforms": [ 
		      # augmentations ...
              {"type": "clip", "round": 16},	# feature stride, see below
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
	# params_weight is of shape (,,,priors)
```

Note that we still need to rasterize any JSON-based annotation even
though we do not need them here, as PicPac is not able to encode
JSON strings into a minibatch.

## Streaming for Mask-RCNN

This is one step further on top of box regression (`box_feature`
transformation and `use_tag` of `rasterize`).

See https://github.com/aaalgo/box/blob/master/train.py for a full
example.

```
import picpac

is_training = True

config = {
          # ... same as box regression ...
          "transforms": [ 
		      # augmentations ...
              {"type": "clip", "round": 16},	# feature stride, see below
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
	# box_feature is of shape (N, 6), where N is the number of boxes
	#    box_feature[:, 0]    object label
	#	 box_feature[:, 1]    object tag
	#    box_feature[:, 2:4]    (x1, y1), top left coordinate, clipped to image area
	#    box_feature[:, 4:6]	(x2, y2), bottom right coordinate, clipped to image area
```




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

