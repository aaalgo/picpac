import picpac
import mxnet
from mxnet.io import DataIter

class ImageIter(DataIter):
    def __init__ (self, path, **kwargs):
        super(ImageIter, self).__init__()
        self.stream = picpac.ImageStream(path, loop=False, **kwargs)
        self.batch_size = kwargs.get('batch', 1)
        self.data = None
        self.label = None
        self.pad = None
        self.peek = self.stream.next()
        self.provide_data = [('data',self.peek[0].shape)]
        self.provide_label = [('softmax_label',self.peek[1].shape)]
        pass

    def reset (self):
        self.stream.reset()
        pass

    def iter_next (self):
        if self.peek:
            self.data, self.label, self.pad = self.peek
            self.peek = None
        else:
            self.data, self.label, self.pad = self.stream.next()
        return True

    def getdata (self):
        return [mxnet.nd.array(self.data)]

    def getlabel (self):
        return [mxnet.nd.array(self.label)]

    def getpad (self):
        return self.pad

