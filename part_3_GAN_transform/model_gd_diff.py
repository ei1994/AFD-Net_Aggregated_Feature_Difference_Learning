# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:03:33 2018

@author: Administrator
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
        
#    U_Net结构， 用于将 nir ---> gray
def create_generator(generator_inputs, generator_outputs_channels, name="generator", reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        layers = []
        with tf.variable_scope("encoder_1"):
            # encoder_1: [batch, 64, 64, 1 ] => [batch, 32, 32, 32]
            convolved = conv(generator_inputs, 32, 3, pad='SAME')
            output = tf.nn.max_pool(convolved,[1,2,2,1],[1,2,2,1],padding='SAME') 
            layers.append(output)
    
        layer_specs = [
            64, # encoder_2: [batch, 32, 32, 32 ] => [batch, 16, 16, 64]
            128, # encoder_3: [batch, 16, 16, 64] => [batch, 8, 8, 128]
            256, # encoder_3: [batch, 8, 8, 128] => [batch, 4, 4, 256]
        ]
    
        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                output = lrelu(layers[-1], 0.2)
                convolved = conv(output, out_channels, 3, pad='SAME')
                output = tf.nn.max_pool(convolved,[1,2,2,1],[1,2,2,1],padding='SAME') 
                output = batchnorm(output)
                layers.append(output)   # 在每一层激活前，进行特征图的保存
    
        layer_specs = [
            (128, 0.0),   # decoder_2: [batch, 4, 4, 256] => [batch, 8, 8, 128]
            (64, 0.0),   # decoder_3: [batch, 8, 8, 128] => [batch, 16, 16, 64]
            (32, 0.0),   # decoder_3: [batch, 16, 16, 64] => [batch, 32, 32, 32]
        ]
        num_encoder_layers = len(layers)  
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    inputs = layers[-1]
                else:
                    inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)  #跨层连接
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                inputs = tf.nn.relu(inputs)
                output = deconv(inputs, out_channels, 3, pad='SAME')
                output = batchnorm(output)
    
                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
    
                layers.append(output)
    
        # decoder_4: [batch, 32, 32, ngf ] => [batch, 64, 64, generator_outputs_channels]
        
        with tf.variable_scope("decoder_1"):
            inputs = tf.concat([layers[-1], layers[0]], axis=3)
            inputs = tf.nn.relu(inputs)
            output = deconv(inputs, generator_outputs_channels,3,pad='SAME')
            output = tf.tanh(output)  # output (-1,1)
            layers.append(output)
        return layers[-1]


# 判别器网络，使用特征图相减求差，替代之前的双通道
def features(img, name, reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        layer1 = conv(img, 32, 3, pad='SAME')                 # 64*64*32
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
#        4*4*256
        return pool4, [layer1, pool1, pool2, pool3, pool4]

def matching(feature_map_1,feature_map_2): 
    with tf.variable_scope('match_layers'):

        diff_feature = tf.abs(feature_map_1 - feature_map_2)  # 2*2*256
        layer5 = conv(diff_feature, 512, 3, pad='SAME')    
        layer5 = tf.nn.relu(layer5)
        pool5 = tf.nn.max_pool(layer5,[1,2,2,1],[1,2,2,1],padding='SAME') # 2*2*512
        
        batch_size = int(pool5.get_shape()[0])
        dense = tf.reshape(pool5, [batch_size,-1])  # reshape as a tensor 2048
        dense1 = tf.layers.dense(dense, 1024)
        dense1 = tf.nn.relu(dense1)
        dense2 = tf.layers.dense(dense1, 256)
        dense2 = tf.nn.relu(dense2)
        dense3 = tf.layers.dense(dense2, 1)
        output = tf.sigmoid(dense3)   
        return output
    
# 判别器网络，p1：gray   p2： nir
def create_discriminator(inputs_p1, inputs_p2, name="discriminator", reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        feature_map_1, layers_1 = features(inputs_p1, 'feature_layers') # 4*4*256
        feature_map_2, layers_2 = features(inputs_p2, 'feature_layers',reuse=True) # 4*4*256
        output = matching(feature_map_1, feature_map_2)   
        return output

def gd_model_g2r(inputs_g, inputs_r):
    # gray --> nir
        out_channels = int(inputs_r.get_shape()[-1])
        g_outputs = create_generator(inputs_g, out_channels)  # fake_optical
        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            # 2x [batch, height, width, channels] => [batch, 1]
            predict_real = create_discriminator(inputs_g, inputs_r)
        with tf.name_scope("fake_discriminator"):
            # 2x [batch, height, width, channels] => [batch, 1]
            predict_fake = create_discriminator(inputs_g, g_outputs, reuse=True)
    
        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
    
        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(inputs_r - g_outputs))
            gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight
            
        return  gen_loss, discrim_loss, g_outputs, [gen_loss_GAN, gen_loss_L1]     
        




