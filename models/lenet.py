import tensorflow as tf
from helpers import *
import numpy as np

class LeNet:
    
    def __init__(self, n_classes=1000, only_features=False, feature_dim=256):
        self.n_classes=n_classes
        self.weights = {}
        self.model=None
        self.only_features=only_features
        self.feature_dim=feature_dim
        
    def build(self, X):
        if not self.only_features:
            self.keep_ratio = tf.placeholder(tf.float32)
        
        model = self.conv2d_relu(X, 5, 20, "conv1")
        model = max_pool2d(model)
        model = self.conv2d_relu(model, 5, 50, "conv2")
        model = max_pool2d(model)
        model = tf.reshape(model, [int(model.get_shape()[0]), -1])
        model = self.dense_relu(model, 500, "fc1")
        
        if not self.only_features:
            model = dropout(model, self.keep_ratio)
            model = self.dense(model, self.n_classes, "logits")

        self.model = model
        self.print_net()
        return self
    
    def stream(self, X, weights=None):
        if weights is None:
            weights=self.weights
        
        model = conv2d_relu(X, weights["conv1"])
        model = max_pool2d(model)
        model = conv2d_relu(model, weights["conv2"])
        model = max_pool2d(model)
        model = tf.reshape(model, [int(model.get_shape()[0]), -1])
        model = dense_relu(model, weights["fc1"])
        
        return model
         
    def get_dropout_ratio(self):
        return self.keep_ratio
    
    def conv2d_relu(self, x, ksize, csize, name, only_weight=False):
        if name in self.weights:
            raise Exception('Name of the filter "{}" already in use.'.format(name))
        _,_,_,cdim = x.get_shape()
        self.weights[name] = tf.Variable(xavier_init([ksize, ksize, int(cdim), csize]), name=name)
        if not only_weight:
            return conv2d_relu(x, self.weights[name])
    
    def dense(self, x, csize, name, only_weight=False):
        if name in self.weights:
            raise Exception('Name of the filter "{}" already in use.'.format(name))
        _,cdim = x.get_shape()
        self.weights[name] = tf.Variable(xavier_init([int(cdim), csize]), name=name)
        if not only_weight:
            return dense(x, self.weights[name])
    
    def dense_relu(self, x, csize, name, only_weight=False):
        op = self.dense(x, csize, name)
        if not only_weight:
            return tf.nn.relu(op)
    
    def get_weight(self, name):
        if not name in self.weights:
            raise Exception('Filter with name "{}" does not exist.'.format(name))
        return self.weights[name]
    
    def get_weights(self, keys=False):
        if keys:
            return list(zip(self.weights.keys(), self.weights.values()))
        return list(self.weights.values())
    
    def print_net(self):
        print("name\tshape")
        print("------------------")
        for k in self.weights.keys():
            print("{}\t{}".format(k, self.weights[k].get_shape()))
            