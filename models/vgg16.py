import tensorflow as tf
from helpers import *
import numpy as np

class VGG16:
    
    def __init__(self, n_classes=1000):
        self.n_classes=n_classes
        self.weights = {}
        self.model=None
        
    def build(self, X):
        ''' 
            Builds a VGG16 model and returns its logits 
            TODO: subtract rgb means from input image
        '''
        self.keep_ratio = tf.placeholder(tf.float32)
        
        model = self.conv2d_relu(X, 3, 64, "conv1a")
        model = self.conv2d_relu(model, 3, 64, "conv1b")
        model = max_pool2d(model)
        
        model = self.conv2d_relu(model, 3, 128, "conv2a")
        model = self.conv2d_relu(model, 3, 128, "conv2b")
        model = max_pool2d(model)
        
        model = self.conv2d_relu(model, 3, 256, "conv3a")
        model = self.conv2d_relu(model, 3, 256, "conv3b")
        model = self.conv2d_relu(model, 3, 256, "conv3c")
        model = max_pool2d(model)
        
        model = self.conv2d_relu(model, 3, 512, "conv4a")
        model = self.conv2d_relu(model, 3, 512, "conv4b")
        model = self.conv2d_relu(model, 3, 512, "conv4c")
        model = max_pool2d(model)
        
        model = self.conv2d_relu(model, 3, 512, "conv5a")
        model = self.conv2d_relu(model, 3, 512, "conv5b")
        model = self.conv2d_relu(model, 3, 512, "conv5c")
        model = max_pool2d(model)
        
        model = tf.reshape(model, [int(model.get_shape()[0]), -1])
        model = self.dense_relu(model, 4096, "fc1")
        model = dropout(model, self.keep_ratio)
        model = self.dense_relu(model, 4096, "fc2")
        model = dropout(model, self.keep_ratio)
        model = self.dense(model, self.n_classes, "logits")
        
        self.model = model
        self.print_net()
        return model
    
    def conv2d_relu(self, x, ksize, csize, name):
        if name in self.weights:
            raise Exception('Name of the filter "{}" already in use.'.format(name))
        _,_,_,cdim = x.get_shape()
#         self.weights[name] = tf.Variable(xavier_init([ksize, ksize, int(cdim), csize]), name=name)
        self.weights[name] = tf.get_variable(name, shape=[ksize, ksize, int(cdim), csize],
                                             initializer=tf.contrib.layers.xavier_initializer())
        return conv2d_relu(x, self.weights[name])
    
    def dense(self, x, csize, name):
        if name in self.weights:
            raise Exception('Name of the filter "{}" already in use.'.format(name))
        _,cdim = x.get_shape()
#         self.weights[name] = tf.Variable(xavier_init([int(cdim), csize]), name=name)
        self.weights[name] = tf.get_variable(name, shape=[int(cdim), csize],
                                             initializer=tf.contrib.layers.xavier_initializer())
        return dense(x, self.weights[name])
    
    def dense_relu(self, x, csize, name):
        return tf.nn.relu(self.dense(x, csize, name))
    
    def get_weight(self, name):
        if not name in self.weights:
            raise Exception('Filter with name "{}" does not exist.'.format(name))
        return self.weights[name]
    
    def get_weights(self, except_=None, only_=None):
        rl = []
        if not except_ is None:
            for k in self.weights.keys():
                if not k in except_:
                    rl.append(self.weights[k])
            return rl
        elif not only_ is None:
            for k in only_:
                rl.append(self.get_weight(k))
            return rl
        
        return list(self.weights.values())
    
    def get_dropout_ratio(self):
        return self.keep_ratio
    
    def print_net(self):
        print("name\tshape")
        print("------------------")
        for k in self.weights.keys():
            print("{}\t{}".format(k, self.weights[k].get_shape()))
            