#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pkgutil
import os
import datetime
import picpac
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from tensorflow.python.client import timeline

# --net=module.model
# where module is the python file basename of one of these
#   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim/nets
# and model is the function name within the file defining the net.
# e.g. the following have worked for very small datasets
#   alexnet.alexnet_v2
#   inception_v3.inception_v3
#   vgg.vgg_16

# Failed to converge:
#   vgg.vgg_a
#   inception_v1.inception_v1
#
# Not working yet:  resnet for requirement of special API (blocks)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'db', 'database')
flags.DEFINE_string('test_db', None, 'evaluation dataset')
flags.DEFINE_integer('classes', '2', 'number of classes')
flags.DEFINE_integer('resize', '224', '')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('batch', 32, 'Batch size.  ')
flags.DEFINE_string('net', 'vgg.vgg_16', 'cnn architecture, e.g. vgg.vgg_a')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('test_steps', 0, 'Number of steps to run evaluation.')
flags.DEFINE_integer('save_steps', 1000, 'Number of steps to run evaluation.')
flags.DEFINE_integer('max_steps', 200000, 'Number of steps to run trainer.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_integer('split', 1, 'split into this number of parts for cross-validation')
flags.DEFINE_integer('split_fold', 0, 'part index for cross-validation')

# load network architecture by name
def inference (inputs, num_classes):
    full = 'tensorflow.contrib.slim.python.slim.nets.' + FLAGS.net
    # e.g. full == 'tensorflow.contrib.slim.python.slim.nets.vgg.vgg_a'
    fs = full.split('.')
    loader = pkgutil.find_loader('.'.join(fs[:-1]))
    module = loader.load_module('')
    net = getattr(module, fs[-1])
    return net(inputs, num_classes)

def fcn_loss (logits, labels):
    with tf.name_scope('loss'):
        labels = tf.to_int32(labels)    # float from picpac
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
        hit = tf.cast(tf.nn.in_top_k(logits, labels, 1, name="accuracy"), tf.float32)
        return [tf.reduce_mean(xe, name='xentropy_mean'), tf.reduce_mean(hit, name='accuracy_total')]
    pass

def training (loss, rate):
    tf.scalar_summary(loss.op.name, loss)
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
                #mixin="db0",
                #mixin_group_delta=0,
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
        X = tf.placeholder(tf.float32, shape=(FLAGS.batch, FLAGS.resize, FLAGS.resize, FLAGS.channels), name="images")
        Y_ = tf.placeholder(tf.float32, shape=(FLAGS.batch,), name="labels")
        logits, _ = inference(X, FLAGS.classes)

        loss, accuracy = fcn_loss(logits, Y_)
        train_op = training(loss, FLAGS.learning_rate)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, tf.get_default_graph())

        init = tf.global_variables_initializer()

        graph_txt = tf.get_default_graph().as_graph_def().SerializeToString()
        with open(os.path.join(FLAGS.train_dir, "graph"), "w") as f:
            f.write(graph_txt)
            pass

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        loss_sum = 0
        accuracy_sum = 0
        batch_sum = 0
        with tf.Session(config=config) as sess:
            sess.run(init)
            for step in xrange(FLAGS.max_steps):
                images, labels, pad = tr_stream.next()
                #print(images.shape, labels.shape)
                feed_dict = {X: images,
                             Y_: labels}
                #l_v, s_v = sess.run([logits, score], feed_dict=feed_dict)
                #print(images.shape, s_v.shape, l_v.shape)
                #_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
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
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
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
                        feed_dict = {X: images,
                                     Y_: labels}
                        _, loss_value, accuracy_value = sess.run([loss, accuracy], feed_dict=feed_dict)
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

