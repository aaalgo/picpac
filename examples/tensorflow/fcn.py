#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import picpac


BATCH = 1

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 200000, 'Number of steps to run trainer.')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('out_channels', 2, '')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

def cp_layer (bottom, scope, params, ksize, kstride, psize, pstride, ch_in, ch_out, relu=True):
    with tf.name_scope(scope):
        filters = tf.Variable(
                        tf.truncated_normal(
                            [ksize, ksize, ch_in, ch_out],
                            dtype=tf.float32,
                            stddev=0.01),
                        name='filters')
        out = tf.nn.conv2d(bottom, filters, [1,kstride,kstride,1], padding="SAME")
        biases = tf.Variable(
                        tf.constant(0.0, shape=[ch_out], dtype=tf.float32),
                        trainable=True,
                        name='bias')
        out = tf.nn.bias_add(out, biases)
        if relu:
            out = tf.nn.relu(out, name=scope)
        if not psize is None:
            out = tf.nn.max_pool(out, ksize=[1,psize,psize,1],
                        strides=[1,pstride,pstride,1],
                        padding='SAME',
                        name='pool')
        params.extend([filters, biases])
        return out
    pass

def inference (images, train=True):
    params = []
    out = cp_layer(images, "layer1", params, 5, 2, 2, 2, FLAGS.channels, 100)
    out = cp_layer(out, "layer2", params, 5, 2, 2, 2, 100, 200)
    out = cp_layer(out, "layer2", params, 3, 1, None, None, 200, 300)
    out = cp_layer(out, "layer3", params, 3, 1, None, None, 300, 300)
    if train:
        out = tf.nn.dropout(out, 0.1, name='dropout')
    out = cp_layer(out, "score", params, 1, 1, None, None, 300, FLAGS.out_channels, relu=False)
    score = out
    with tf.name_scope('upscale'):
        shape = tf.unpack(tf.shape(images))
        print(shape.__class__)
        shape.pop()
        shape.append(tf.constant(FLAGS.out_channels, dtype=tf.int32))
        print(len(shape))
        filters = tf.Variable(
                        tf.truncated_normal(
                            [31, 31, FLAGS.out_channels, FLAGS.out_channels],
                            dtype=tf.float32,
                            stddev=0.01),
                        name='filters')
        logits = tf.nn.conv2d_transpose(out, filters, tf.pack(shape),
                        [1,16,16,1], padding='SAME', name='upscale')
        # do we want to add bias?
    return logits, score, params

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
    db='db'
    tr_stream = picpac.ImageStream(db, negate=False, perturb=True, **config)

    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, shape=(BATCH, None, None, FLAGS.channels), name="images")
        Y_ = tf.placeholder(tf.int32, shape=(BATCH, None, None, 1), name="labels")
        logits, score, params = inference(X)
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

