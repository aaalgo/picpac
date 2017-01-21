#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pkgutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import datetime
import picpac
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils

def resnet_v1_slim_fc (inputs, scope):
    """ slim version of resnet, can be replaced with
        
        resnet_v1.resnet_v1_50(inputs, num_classes=None, global_pool=True,
                               output_stride=None, reuse=False, scope=scope)

        or any of the other resnsets
    """
    blocks = [
      resnet_utils.Block('block1', resnet_v1.bottleneck,
                         [(64, 32, 1)] * 2 + [(64, 32, 2)]),
      resnet_utils.Block('block2', resnet_v1.bottleneck,
                         [(128, 64, 1)] * 3 + [(128, 64, 2)]),
      resnet_utils.Block('block3', resnet_v1.bottleneck,
                         [(256, 64, 1)] * 3 + [(256, 64, 2)]),
      resnet_utils.Block('block4', resnet_v1.bottleneck, [(128, 64, 1)] * 1)
    ]
    return resnet_v1.resnet_v1(
      inputs, blocks,
      # all parameters below can be passed to resnet_v1.resnet_v1_??
      num_classes = None,       # don't produce final prediction
      global_pool = True,       # produce 1x1 output, equivalent to input of a FC layer
      output_stride = None,
      include_root_block=True,
      reuse=False,              # do not re-use network
                                # my understanding
                                # task1      image -> resnet1 -> output
                                # task2      image -> resnet2 -> output
                                # if both resnets are defined under the same scope,
                                # with reuse set to True, then some of the parameters
                                # will be shared between two tasks
      scope=scope)

#####################################################################################
def multipod (inputs, num_classes):
    branches = []
    with tf.name_scope('multipod'):
        for i, input in enumerate(inputs):
            # create a net for each input, we can do each branch differently
            branch, _ = resnet_v1_slim_fc(input, scope = 'branch%d' % i)
            # 
            # branch is of the shape [?, 1, 1, ?]
            branches.append(branch)
        # concatenate along the last dimension
        net = tf.concat(3, branches)
        # shape is [?, 1, 1, ? * 3], e.g. 256 * 3
        # [1,1] convolution is equivalent to fully-connected layers
        #net = slim.conv2d(net, 256, [1, 1], scope='fc1')
        #net = slim.conv2d(net, 256, [1, 1], scope='fc2')
        net = slim.conv2d(net, num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            scope='fc3')
        # shape is [?, 1, 1, ?], remove the size-1 dimensions
        net = tf.squeeze(net, axis=[1,2])
    return net

def make_feed_dict (Xs, Y, images, labels):
    # we validate the multipod concept by splitting input images
    # into N parts, each fed into a CNN, and merge the output
    # features to predict the final labels

    # if images are from multiple sources we can do
    # return {Y:labels, X[0]:images0, X[1], images1, ...}
    N = len(Xs)
    assert images.shape[2] % N == 0
    step = images.shape[2] // N  # split horizontally

    fd = {Y: labels}
    for i, X in enumerate(Xs):
        block = images[:,:,i*step:(i+1)*step,:]
        fd[X] = block
        pass
    return fd
############################################################################


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'db', 'database')
flags.DEFINE_string('test_db', None, 'evaluation dataset')
flags.DEFINE_integer('classes', '2', 'number of classes')
flags.DEFINE_integer('resize', '224', '')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('branches', 3, '')
flags.DEFINE_integer('disable', None, 'disable this branch')
flags.DEFINE_integer('batch', 32, 'Batch size.  ')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('test_steps', 0, 'Number of steps to run evaluation.')
flags.DEFINE_integer('save_steps', 1000, 'Number of steps to run evaluation.')
flags.DEFINE_integer('max_steps', 200000, 'Number of steps to run trainer.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_integer('split', 1, 'split into this number of parts for cross-validation')
flags.DEFINE_integer('split_fold', 0, 'part index for cross-validation')

def fcn_loss (logits, labels):
    with tf.name_scope('loss'):
        labels = tf.to_int32(labels)    # float from picpac
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
        hit = tf.cast(tf.nn.in_top_k(logits, labels, 1, name="accuracy"), tf.float32)
        return [tf.reduce_mean(xe, name='xentropy_mean'), tf.reduce_mean(hit, name='accuracy_total')]
    pass

def training (loss, rate):
    optimizer = tf.train.GradientDescentOptimizer(rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return optimizer.minimize(loss, global_step=global_step)

def run_training ():
    config = dict(seed=1996,
                shuffle=True,
                reshuffle=True,
                resize_width=FLAGS.resize,
                resize_height=FLAGS.resize,
                batch=FLAGS.batch,
                split=FLAGS.split,
                split_fold=FLAGS.split_fold,
                channels=FLAGS.channels,
                stratify=True,
                pert_color1=10,
                pert_color2=10,
                pert_color3=10,
                pert_angle=10,
                pert_min_scale=0.8,
                pert_max_scale=1.2,
                #pad=False,
                #pert_hflip=True,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )
    # training stream
    tr_stream = picpac.ImageStream(FLAGS.db, split_negate=False, perturb=True, loop=True, **config)
    te_stream = None
    if FLAGS.test_steps > 0:
        # testing stream, "negate" inverts the image selection specified by split & split_fold
        # so different images are used for training and testing
        if FLAGS.test_db:
            if FLAGS.split > 1:
                print("Cannot use cross-validation & evaluation db at the same time")
                print("If --test-db is specified, do not set --split")
                raise Exception("bad parameters")
            te_stream = picpac.ImageStream(FLAGS.test_db, perturb=False, loop=False, **config)
        elif FLAGS.split > 1:
            te_stream = picpac.ImageStream(FLAGS.db, split_negate=True, perturb=False, loop=False, **config)
            pass
        pass

    with tf.Graph().as_default():
        shape = (FLAGS.batch, FLAGS.resize, FLAGS.resize/FLAGS.branches, FLAGS.channels)
        Xs = [tf.placeholder(tf.float32, shape=shape, name="input%d" % i) for i in range(FLAGS.branches)]
        Y = tf.placeholder(tf.float32, shape=(FLAGS.batch,), name="labels")

        if FLAGS.disable:
            logits = multipod([X for i, X in enumerate(Xs) if i != FLAGS.disable], FLAGS.classes)
        else:
            logits = multipod(Xs, FLAGS.classes)

        loss, accuracy = fcn_loss(logits, Y)
        train_op = training(loss, FLAGS.learning_rate)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        loss_sum = 0
        accuracy_sum = 0
        batch_sum = 0
        with tf.Session(config=config) as sess:
            sess.run(init)
            for step in xrange(FLAGS.max_steps):
                images, labels, pad = tr_stream.next()
                #print(images.shape, labels.shape)
                feed_dict = make_feed_dict(Xs, Y, images, labels)

                _, loss_value, accuracy_value = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
                loss_sum += loss_value * FLAGS.batch
                accuracy_sum += accuracy_value * FLAGS.batch
                batch_sum += FLAGS.batch
                if step % 100 == 0:
                    #tl = timeline.Timeline(run_metadata.step_stats)
                    #ctf = tl.generate_chrome_trace_format()
                    #with open('timeline.json', 'w') as f:
                    #    f.write(ctf)

                    print(datetime.datetime.now())
                    print('step %d: loss = %.4f, accuracy = %.4f' % (step, loss_sum/batch_sum, accuracy_sum/batch_sum))
                    loss_sum = 0
                    accuracy_sum = 0
                    batch_sum = 0
                if te_stream and step % FLAGS.test_steps == 0:
                    # evaluation
                    te_stream.reset()
                    batch_sum2 = 0
                    loss_sum2 = 0
                    accuracy_sum2 = 0
                    for images, labels, pad in te_stream:
                        bs = FLAGS.batch - pad
                        if pad > 0:
                            numpy.resize(images, (bs,)+images.shape[1:])
                            numpy.resize(labels, (bs,))
                        feed_dict = make_feed_dict(Xs, Y, images, labels)
                        _, loss_value, accuracy_value = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
                        batch_sum2 += bs
                        loss_sum2 += loss_value * bs
                        accuracy_sum2 += accuracy_value * bs
                        pass
                    print('evaluation: loss = %.4f, accuracy = %.4f' % (loss_sum2/batch_sum2, accuracy_sum2/batch_sum2))
                if (step + 1) % FLAGS.save_steps == 0 or (step + 1) == FLAGS.max_steps:
                    saver.save(sess, os.path.join(FLAGS.train_dir, "model"), global_step=step)
                pass
            pass
        pass
    pass


def main (_):
    run_training()

if __name__ == '__main__':
    tf.app.run()

