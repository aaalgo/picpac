#!/usr/bin/env python
'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import os
import sys
import time
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import picpac

batch_size = 32
nb_classes = 10
nb_epoch = 20
data_augmentation = True

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def check_download(filename, source='http://www.aaalgo.com/picpac/datasets/cifar/'):
        if not os.path.exists(filename):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)

    config = dict(seed=1996,
                resize_width=32,
                resize_height=32,
                batch=batch_size,
                channels=3,
                channel_first=False,
                onehot=nb_classes
                )
    train_config = dict(
                shuffle=True,
                reshuffle=True,
                split=10,
                split_fold=1,
                stratify=True,
                pert_color1=10,
                pert_color2=10,
                pert_color3=10,
                pert_angle=15,
                pert_min_scale=0.7,
                pert_max_scale=1.2,
                )
    train_config.update(config)

    # We can now download and read the training and test set images and labels.
    check_download('cifar10-train.picpac')
    check_download('cifar10-test.picpac')
    train = picpac.ImageStream('cifar10-train.picpac', loop=True, split_negate=False, perturb=data_augmentation, **train_config)
    val = picpac.ImageStream('cifar10-train.picpac', loop=False, split_negate=True, perturb=False, **train_config)
    test = picpac.ImageStream('cifar10-test.picpac', loop=False, perturb=False, **config)

    return train, val, test


model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_str, val_str, test_str = load_dataset()

epoch_size = train_str.size() / batch_size
for epoch in range(nb_epoch):
    # In each epoch, we do a full pass over the training data:
    train_loss = 0
    train_acc = 0
    train_batches = 0
    start_time = time.time()
    for _ in tqdm(range(epoch_size), leave=False):
        inputs, targets, _ = train_str.next()
        inputs /= 255
        loss, acc = model.train_on_batch(inputs, targets)
        train_loss += loss
        train_acc += acc
        train_batches += 1

    # And a full pass over the validation data:
    val_loss = 0
    val_acc = 0
    val_batches = 0
    test_str.reset() # re-start streaming
    for inputs, targets, pad in val_str:
        if pad: # not full batch
            break
        inputs /= 255
        loss, acc = model.test_on_batch(inputs, targets)
        val_loss += loss
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {}/{} time {:.3f}s; train loss: {:.6f} acc:{:.6f}; val loss: {:.6f} acc:{:.2f}".format(
        epoch + 1, nb_epoch, time.time() - start_time,
        train_loss/train_batches, train_acc/train_batches,
        val_loss/val_batches, val_acc/val_batches))

# After training, we compute and print the test error:
test_loss = 0
test_acc = 0
test_batches = 0
for inputs, targets, pad in test_str:
    if pad: # not full batch
        break
    inputs /= 255
    loss, acc = model.test_on_batch(inputs, targets)
    test_loss += loss
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss: {:.6f} acc: {:.2f}".format(test_loss/test_batches, test_acc/test_batches))

