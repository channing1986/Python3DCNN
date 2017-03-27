import sys, os
import tensorflow as tf
import math
import numpy as np



class Vgg16Model():
    def __init__(self):
        self.image_mean = np.array([103.939, 116.779, 123.68])

    def conv2d(self,x, n_in, n_out, k, s, p='SAME', bias=True, scope=None):
        with tf.variable_scope(scope or 'Conv2D') as scope:
            kernel_init_std = math.sqrt(2.0 / (k * k * n_in))
            kernel = tf.get_variable('Weight', shape=[k,k,n_in,n_out],
            initializer=tf.truncated_normal_initializer(0.0, kernel_init_std))
            tf.add_to_collection('Weights', kernel)
            y = tf.nn.conv2d(x, kernel, [1,1,s,s], padding=p, data_format='NCHW')
            if bias == True:
                bias = tf.get_variable('Bias', shape=[n_out],
                initializer=tf.constant_initializer(0.0))
                tf.add_to_collection('Biases', bias)
                y = tf.nn.bias_add(y, bias, data_format='NCHW')
        return y

    def linear(self,x, n_in, n_out, bias=True, scope=None):
      with tf.variable_scope(scope or 'Linear'):
        weight_init_std = math.sqrt(1.0 / n_out)
        weight = tf.get_variable('Weight', shape=[n_in,n_out],
          initializer=tf.truncated_normal_initializer(0.0, weight_init_std))
        tf.add_to_collection('Weights', weight)
        y = tf.matmul(x, weight)
        if bias == True:
          bias = tf.get_variable('Bias', shape=[n_out],
            initializer=tf.constant_initializer(0.0))
          tf.add_to_collection('Biases', bias)
          y = y + bias
      return y

    def mlp(self,x, n_in, n_hidden, n_out, activation=tf.nn.relu, scope=None):
      with tf.variable_scope(scope or 'Mlp'):
        y = self.linear(x, n_in, n_hidden, scope='Linear1')
        y = activation(y)
        y = self.linear(y, n_hidden, n_out, scope='Linear2')
      return y


    def _vgg_conv_relu(self, x, n_in, n_out, scope):
        with tf.variable_scope(scope):
            conv = self.conv2d(x, n_in, n_out, 3, 1,'SAME')
            relu = tf.nn.relu(conv)
        return relu

    def _vgg_max_pool(self, x, scope):
        with tf.variable_scope(scope):
            pool = tf.nn.max_pool(x, [1,1,2,2], [1,1,2,2],
            padding='SAME', data_format='NCHW')
        return pool

    def _vgg_fully_connected(self, x, n_in, n_out, scope):
        with tf.variable_scope(scope):
            fc = self.linear(x, n_in, n_out)
        return fc

    def __call__(self, x, scope=None):
        with tf.variable_scope(scope or 'Vgg16'):
          # conv stage 1
          relu1_1 = self._vgg_conv_relu(x, 3, 64, 'conv1_1')
          relu1_2 = self._vgg_conv_relu(relu1_1, 64, 64, 'conv1_2')
          pool1 = self._vgg_max_pool(relu1_2, 'pool1')
          # conv stage 2
          relu2_1 = self._vgg_conv_relu(pool1, 64, 128, 'conv2_1')
          relu2_2 = self._vgg_conv_relu(relu2_1, 128, 128, 'conv2_2')
          pool2 = self._vgg_max_pool(relu2_2, 'pool2')
          # conv stage 3
          relu3_1 = self._vgg_conv_relu(pool2, 128, 256, 'conv3_1')
          relu3_2 = self._vgg_conv_relu(relu3_1, 256, 256, 'conv3_2')
          relu3_3 = self._vgg_conv_relu(relu3_2, 256, 256, 'conv3_3')
          pool3 = self._vgg_max_pool(relu3_3, 'pool3')
          # conv stage 4
          relu4_1 = self._vgg_conv_relu(pool3, 256, 512, 'conv4_1')
          relu4_2 = self._vgg_conv_relu(relu4_1, 512, 512, 'conv4_2')
          relu4_3 = self._vgg_conv_relu(relu4_2, 512, 512, 'conv4_3')
          pool4 = self._vgg_max_pool(relu4_3, 'pool4')
          # conv stage 5
          relu5_1 = self._vgg_conv_relu(pool4, 512, 512, 'conv5_1')
          relu5_2 = self._vgg_conv_relu(relu5_1, 512, 512, 'conv5_2')
          relu5_3 = self._vgg_conv_relu(relu5_2, 512, 512, 'conv5_3')
          pool5 = self._vgg_max_pool(relu5_3, 'pool5')
          # fc6
          n_conv_out = 7*7*512
          flatten = tf.reshape(pool5, [-1,n_conv_out])
          fc6 = self._vgg_fully_connected(flatten, n_conv_out, 4096, scope='fc6')
          relu_6 = tf.nn.relu(fc6)
          # fc7
          fc7 = self._vgg_fully_connected(fc6, 4096, 4096, scope='fc7')
          relu_7 = tf.nn.relu(fc7)
          # fc8, prob
          fc8 = self._vgg_fully_connected(relu_7, 4096, 1000, scope='fc8')
          prob = tf.nn.softmax(fc8)
        return prob