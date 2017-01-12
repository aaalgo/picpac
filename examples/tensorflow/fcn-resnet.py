#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
# RESNET: import these for slim version of resnet
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils

import picpac

BATCH = 1

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', None, '')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 200000, 'Number of steps to run trainer.')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('out_channels', 2, '')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

# RESNET
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v1.py
# reset_v1.reset_v1 is meta function that generates actual
# ResNet architectures.  The architecture generated can be configured
# by the "blocks" argument, which is a list of construction units.
# The end architectures resnet_v1 provides, i.e. resnet_v1.restnet_v1_50/101/152/200
# differ only by the number and configuration of blocks.

# Here we use this meta function to construct a slim version of 
# resnet which trains faster.
def resnet_v1_slim (inputs,
                  num_classes=None,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,   # the above parameters will be directly passed to
                                # resnet.resnet_v1
                  scope='resnet_v1_slim'):
  blocks = [
      resnet_utils.Block('block1', resnet_v1.bottleneck,
                         [(64, 32, 1)] * 2 + [(64, 32, 2)]),
      # the last argument of Block is a list of "bottleneck" unit
      # configurations. Each entry is of the form  [depth, in-depth, stride]
      # Each "bottleneck" unit consists 3 layers:
      #    convolution from depth channels to in-depth channels
      #    convolution from in-depth channels to in-depth channels
      #    convolution from in-depth channels to depth channels
      # It's called "bottleneck" because the overall input and output
      # depth (# channels) are the same, while the in-depth in the 
      # middle is smaller.

      # Because each bottleneck has 3 layers, the above chain has
      # 3 * (2 + 1) = 9 layers.

      # By convention alll bottleneck units have stride = 1 except for the last which has
      # stride of 2.  The overall effect is after the whole chain, image size
      # is reduced by 2.

      # The original resnet implementation has:
      #   -- very long chains
      #   -- very large depth and in-depth values.
      # This is necessary for very big datasets like ImageNet, but for
      # smaller and simpler datasets we should be able to substantially
      # reduce these, as is what we do in this resnet_slim
      # 
      resnet_utils.Block('block2', resnet_v1.bottleneck,
                         [(128, 64, 1)] * 4 + [(128, 64, 2)]),
      # 3 * (4+1) = 15 layers
      resnet_utils.Block('block3', resnet_v1.bottleneck,
                         [(256, 64, 1)] * 4 + [(256, 64, 2)]),
      # 3 * (4+1) = 15 layers
      resnet_utils.Block('block4', resnet_v1.bottleneck, [(256, 64, 1)] * 2)
      # 3 * 2 = 6 layers
      # so we have  9 + 15 + 15 + 6 = 45 layers
      # there are two extra layers added by the system, so
      # by the reset nomenclature this network can be called a reset_v1_47
      
      # The first 3 Blocks each have stride = 2, and last Block is 1,
      # so the overall stride of this architecture is 8

      # If "output_stride" is smaller than 8, resnet_v1.resnet_v1
      # will add extra down-sizing layers to meet the requirement.
  ]
  return resnet_v1.resnet_v1(
      inputs,
      blocks,
      num_classes,
      global_pool,
      output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)


def inference (images, train=True, resnet_stride=8):
    with slim.arg_scope(resnet_v1.resnet_arg_scope(train)):
        net, end_points = resnet_v1_slim(images,
                                num_classes = None,
                                global_pool = False,
                                output_stride = resnet_stride)
        # replace resnet_v1_slim above with resnet_v1.resnet_v1_50/101/...
        # to use standard architectures.

    #  num_classes: Number of predicted classes for classification tasks. If None
    #      we return the features before the logit layer.
    # global_pool: If True, we perform global average pooling before computing the
    #      logits. Set to True for image classification, False for dense prediction.
    # output_stride: If None, then the output will be computed at the nominal
    #      network stride. If output_stride is not None, it specifies the requested
    #      ratio of input to output spatial resolution.
    resnet_depth = utils.last_dimension(net.get_shape(), min_rank=4)

    shape = tf.unpack(tf.shape(images))
    print(shape.__class__)
    shape.pop()
    shape.append(tf.constant(FLAGS.out_channels, dtype=tf.int32))
    print(len(shape))
    filters = tf.Variable(
                    tf.truncated_normal(
                        [resnet_stride*2+1, resnet_stride*2+1, FLAGS.out_channels, resnet_depth],
                        dtype=tf.float32,
                        stddev=0.01),
                    name='filters')
    logits = tf.nn.conv2d_transpose(net, filters, tf.pack(shape),
                    [1,resnet_stride,resnet_stride,1], padding='SAME', name='upscale')
    return logits

def fcn_loss (logits, labels):
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, FLAGS.out_channels))
        labels = tf.to_int32(labels)    # float from picpac
        labels = tf.reshape(labels, (-1,))
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
        return tf.reduce_mean(xe, name='xentropy_mean')
    pass

def training (loss, rate):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return optimizer.minimize(loss, global_step=global_step)

def run_training ():
    seed = 1996
    config = dict(seed=seed,
                loop=True,
                shuffle=True,
                reshuffle=True,
                #resize_width=256,
                #resize_height=256,
                batch=1,
                split=1,
                split_fold=0,
                annotate='json',
                channels=FLAGS.channels,
                stratify=False,
                #mixin="db0",
                #mixin_group_delta=0,
                #pert_color1=10,
                #pert_angle=5,
                #pert_min_scale=0.8,
                #pert_max_scale=1.2,
                #pad=False,
                #pert_hflip=True,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )
    db=FLAGS.db
    tr_stream = picpac.ImageStream(db, negate=False, perturb=True, **config)

    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, shape=(BATCH, None, None, FLAGS.channels), name="images")
        Y_ = tf.placeholder(tf.int32, shape=(BATCH, None, None, 1), name="labels")
        logits = inference(X)
        loss = fcn_loss(logits, Y_)
        train_op = training(loss, FLAGS.learning_rate)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, tf.get_default_graph())

        init = tf.initialize_all_variables()

        graph_txt = tf.get_default_graph().as_graph_def().SerializeToString()
        with open(os.path.join(FLAGS.train_dir, "graph"), "w") as f:
            f.write(graph_txt)
            pass

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            for step in xrange(FLAGS.max_steps):
                images, labels, pad = tr_stream.next()
                #print(images.shape, labels.shape)
                feed_dict = {X: images,
                             Y_: labels}
                #l_v, s_v = sess.run([logits, score], feed_dict=feed_dict)
                #print(images.shape, s_v.shape, l_v.shape)
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                if step % 100 == 0:
                    print('step %d: loss = %.4f' % (step, loss_value))
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    saver.save(sess, os.path.join(FLAGS.train_dir, "model"), global_step=step)
                pass
            pass
        pass
    pass


def main (_):
    run_training()

if __name__ == '__main__':
    tf.app.run()

