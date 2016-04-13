# Streaming Pipeline

# Generic Streaming

# MXNet

```
from picpac.mxnet import ImageStream


# create ImageStream using the same configuration parameters
# picpac.mxnet.ImageStream is inherited from mxnet.io.DataIter

```

# Neon

```
from picpac.neon import ImageStream

# create ImageStream using the same configuration parameters
# picpac.meon.ImageStream is inherited from neon.data.dataiterator.NervanaDataIterator

```

# Caffe

Use this [Caffe FCN fork](https://github.com/aaalgo/caffe-picpac).

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

The default behavior of Caffe is one-hot encoding, so do not specify onehot in configuration.



