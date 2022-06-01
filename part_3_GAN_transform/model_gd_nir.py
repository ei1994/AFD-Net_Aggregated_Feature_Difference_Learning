# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:03:33 2018

@author: Administrator

自编码器，用于 nir 特征编码的学习
"""

import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
a = parser.parse_args()

EPS = 1e-12

def conv(batch_input, out_channels, filter_size, pad, stride=1):
    conv = tf.layers.conv2d(batch_input, out_channels,filter_size,strides=(stride,stride), padding=pad)
    return conv

def deconv(batch_input, out_channels, filter_size, pad):
    conv = tf.layers.conv2d_transpose(batch_input, out_channels, filter_size, strides=(2,2), padding=pad ,kernel_initializer=tf.random_normal_initializer(0, 0.02))
    return conv

def batchnorm(inputs):
    normalized = tf.layers.batch_normalization(inputs, axis=-1)
    return normalized

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return tf.subtract(tf.multiply(image, 2.0),1.0)
#        return image * 2.0 - 1.0

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return tf.div(tf.add(image , 1.0),2.0) 
#        return (image + 1.0) / 2.0

def evaluation(logits, labels):
    correct_prediction = tf.equal(logits, labels)
    correct_batch = tf.reduce_mean(tf.cast(correct_prediction, tf.int32), 1)
    return tf.reduce_sum(tf.cast(correct_batch, tf.float32)), correct_batch
  
# 自编码器，nir图像
def nir_encode(generator_inputs, name="nir_encode", reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        
        layer1 = conv(generator_inputs, 32, 3, pad='SAME')                 # 64*64*32
        layer1 = tf.nn.relu(layer1)
        pool1 = tf.nn.max_pool(layer1,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer2 = conv(pool1, 64, 3, pad='SAME')              # 32*32*64
        layer2 = tf.nn.relu(layer2)
        pool2 = tf.nn.max_pool(layer2,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer3 = conv(pool2, 128, 3, pad='SAME')              # 16*16*128   
        layer3 = tf.nn.relu(layer3)
        pool3 = tf.nn.max_pool(layer3,[1,2,2,1],[1,2,2,1],padding='SAME')
        
        layer4 = conv(pool3, 256, 3, pad='SAME')              # 8*8*256     
        layer4 = tf.nn.relu(layer4)
        pool4 = tf.nn.max_pool(layer4,[1,2,2,1],[1,2,2,1],padding='SAME')
        
        return pool4, [pool1, pool2, pool3, pool4]
  

def nir_decode( nir_encode_out,nir_encode_layers, generator_outputs_channels, name="nir_decode", reuse=False):
    with tf.variable_scope(name,reuse=reuse): 
      
        layer1 = deconv(nir_encode_out, 128, 3, pad='SAME')                # 8*8*128
#        layer1 = batchnorm(layer1)
        layer1_con = tf.concat([layer1, nir_encode_layers[2]], axis=3)
        layer1 = tf.nn.relu(layer1_con)
        
        layer2 = deconv(layer1, 64, 3, pad='SAME')                         # 16*16*64
#        layer2 = batchnorm(layer2)
        layer2_con = tf.concat([layer2, nir_encode_layers[1]], axis=3)
        layer2 = tf.nn.relu(layer2_con)
        
        layer3 = deconv(layer2, 32, 3, pad='SAME')                         # 32*32*32
#        layer3 = batchnorm(layer3)
        layer3_con = tf.concat([layer3, nir_encode_layers[0]], axis=3)
        layer3 = tf.nn.relu(layer3_con)
        
        layer4 = deconv(layer3, generator_outputs_channels, 3, pad='SAME')  # 64*64*1
        output = tf.tanh(layer4)
        
        return output

def create_generator(generator_inputs, generator_outputs_channels, name="generator", reuse=False):
#    with tf.variable_scope(name,reuse=reuse):
        nir_encode_out, nir_encode_layers = nir_encode(generator_inputs)
        output = nir_decode(nir_encode_out, nir_encode_layers, generator_outputs_channels)
        
        return output, nir_encode_layers
    
#  训练 nir --> nir 的自编码器，主要是学习网络对 nir 的编码能力
#  此处的输入和输出图像都是 nir 图像
def gd_model_r2r(inputs_g):
    # nir --> nir
#    with tf.variable_scope("sar2opt"):
        
        out_channels = int(inputs_g.get_shape()[-1])
        g_outputs, gray_layers = create_generator(inputs_g, out_channels)  # fake_optical
        
        with tf.name_scope("generator_loss"):
            gen_loss_L1 = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(inputs_g , g_outputs), 2))
            
#          map_sub = tf.subtract(inputs_r , g_outputs)
#          gen_loss_L1 = 10 * tf.reduce_mean(tf.reduce_mean(map_sub, (1,2,3)))
            
        return  gen_loss_L1, gray_layers
        




