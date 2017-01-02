#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,sys
import cv2
os.environ['LD_LIBRARY_PATH']='/opt/cuda/lib64'
import numpy as np
import tensorflow as tf
import picpac


BATCH = 1

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 200000, 'Number of steps to run trainer.')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('out_channels', 2, '')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

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
    return logits, params

def run_predict (image):
    model = 'data/model-16999'
    with tf.Graph().as_default() as graph:
        saver = tf.train.import_meta_graph(model + '.meta')
        X = graph.get_tensor_by_name("images:0") #tf.placeholder(tf.float32, shape=(BATCH, None, None, FLAGS.channels), name="images")
        logits = graph.get_tensor_by_name("upscale/upscale:0")
        #logits, params = inference(X)
        shape = tf.shape(logits)
        logits = tf.reshape(logits, (-1, FLAGS.out_channels))
        prob = tf.nn.softmax(logits)
        prob = tf.reshape(prob, shape)
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, model)
            prob = sess.run(prob, feed_dict={X: image})
            pass
        pass
    return prob[:,:,:,1]

def main (_):
    image = cv2.imread('input.jpg', -1)
    sz = image.shape[:2]
    image = cv2.resize(image, None, None, 1.7, 1.7)
    shape = (1,)+image.shape
    image = np.reshape(image, shape)
    prob = run_predict(image)
    prob = prob[0]
    prob = cv2.resize(prob, (sz[1], sz[0]))
    print(prob.shape)
    prob *= 255
    cv2.imwrite('output.jpg', prob)

if __name__ == '__main__':
    tf.app.run()

