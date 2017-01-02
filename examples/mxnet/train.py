#!/usr/bin/env python
import sys
import logging
import argparse
import os
import json
import mxnet as mx
from picpac.mxnet import ImageStream

# this is largely based on mxnet's train_model.py
def fit (args, network, data_loader):
    # logging
    head = '%(asctime)-15s %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}
    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    (train, val) = data_loader(args)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args)

    eval_metrics = ['accuracy']

    batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))

    model.fit(
       X                  = train,
       eval_data          = val,
       eval_metric        = eval_metrics,
       kvstore            = None,
       batch_end_callback = batch_end_callback,
       epoch_end_callback = checkpoint)
    pass

# don't use -n and -s, which are resevered for the distributed training
parser = argparse.ArgumentParser(description='train an image classifer on imagenet')
parser.add_argument('--network', type=str, default='inception-bn',
                    choices = ['alexnet', 'vgg', 'googlenet', 'inception-bn', 'inception-bn-full', 'inception-v3'],
                    help = 'the cnn to use')
parser.add_argument('--db', type=str, required=True,
                    help='input dataset')
parser.add_argument('--classes', type=int, required=True,
                    help='the number of classes')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str,
                    help='the prefix of the model to save')
parser.add_argument('--lr', type=float, default=.01,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--clip-gradient', type=float, default=5.,
                    help='clip min/max gradient to prevent extreme value')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--epoch-size', type=int, default=100,
                    help='batches in each epoch')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size')
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--log-file', type=str, 
            help='the name of log file')
parser.add_argument('--log-dir', type=str, default="/tmp/",
                    help='directory of the log file')
parser.add_argument('--data-shape', type=int, default=224,
                    help='set image\'s shape')
args = parser.parse_args()

# network
import importlib
net = importlib.import_module('symbol_' + args.network).get_symbol(args.classes)

# data
def get_iterator(args):

    config = dict(seed=1996,
                loop=True,
                shuffle=True,
                reshuffle=True,
                resize_width=args.data_shape,
                resize_height=args.data_shape,
                batch=args.batch_size,
                split=5,
                split_fold=1,
                channels=3,
                stratify=True,
                pert_color1=10,
                pert_color2=10,
                pert_color3=10,
                pert_angle=10,
                pert_min_scale=0.8,
                pert_max_scale=1.2,
                )
    train = ImageStream(args.db, negate=False, perturb=True, **config)
    val = ImageStream(args.db, negate=True, perturb=False, **config)

    return (train, val)

# train
fit(args, net, get_iterator)

