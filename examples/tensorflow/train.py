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


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('classes', '2', 'number of classes')
flags.DEFINE_integer('resize', '224', '')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('batch', 32, 'Batch size.  ')
flags.DEFINE_string('net', 'vgg.vgg_a', 'cnn architecture, e.g. vgg.vgg_a')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 200000, 'Number of steps to run trainer.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

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
        return tf.reduce_mean(xe, name='xentropy_mean')
    pass

def training (loss, rate):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return optimizer.minimize(loss, global_step=global_step)

def run_training ():
    config = dict(seed=1996,
                loop=True,
                shuffle=True,
                reshuffle=True,
                resize_width=FLAGS.resize,
                resize_height=FLAGS.resize,
                batch=FLAGS.batch,
                split=5,
                split_fold=1,
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
    db='db'
    tr_stream = picpac.ImageStream(db, negate=False, perturb=True, **config)

    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, shape=(FLAGS.batch, FLAGS.resize, FLAGS.resize, FLAGS.channels), name="images")
        Y_ = tf.placeholder(tf.float32, shape=(FLAGS.batch,), name="labels")
        logits, _ = inference(X, FLAGS.classes)

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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
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
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                if step % 100 == 0:
                    #tl = timeline.Timeline(run_metadata.step_stats)
                    #ctf = tl.generate_chrome_trace_format()
                    #with open('timeline.json', 'w') as f:
                    #    f.write(ctf)

                    print(datetime.datetime.now())
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

