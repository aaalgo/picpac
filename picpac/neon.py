from __future__ import absolute_import 
import sys
from operator import mul
import numpy
import picpac
from neon.data.dataiterator import NervanaDataIterator

# Neon use col-major/fortran array format
# so we need to transpose the data.

class ImageStream(NervanaDataIterator):
    def __init__ (self, path, **kwargs):
        super(ImageStream, self).__init__()
        self.stream = picpac.ImageStream(path, **kwargs)
        self.batch_size = kwargs.get('batch', 1)
        self.nclass = kwargs.get('nclass')
        self.train = kwargs.get('train', True)
        self.ndata = self.stream.size()
        self.peek = self.stream.next()
        self.shape = tuple(self.peek[0].shape[1:])
        self.xdim = reduce(mul, self.shape)
        yshape = tuple(self.peek[1].shape[1:])
        if self.nclass:
            assert len(yshape) == 1
        self.ydim = reduce(mul, yshape)
        self.Xbuf = self.be.iobuf(self.xdim)
        self.Ybuf = self.be.iobuf(self.ydim)
        pass

    def reset (self):
        if not self.train:
            self.stream.reset()
        pass

    @property
    def nbatches (self):
        # this behaves differently from neon data loader, which
        # pads the last incomplete batch with data from beginning
        return self.stream.size() // self.batch_size

    def next (self):
        if self.peek:
            X, Y, _ = self.peek
            self.peek = None
        else:
            X, Y, _ = self.stream.next() 
        X = X.reshape(self.batch_size, self.xdim)
        Y = Y.reshape(self.batch_size, self.ydim)
        self.be.copy_transpose(self.be.array(X), self.Xbuf)
        self.be.copy_transpose(self.be.array(Y), self.Ybuf)
        return (self.Xbuf, self.Ybuf)

    def __iter__ (self):
        return self

