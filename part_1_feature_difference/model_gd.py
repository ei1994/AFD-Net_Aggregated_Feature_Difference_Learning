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

def evaluation(logits, labels):
      correct_prediction = tf.equal(logits, labels)
      correct_batch = tf.reduce_mean(tf.cast(correct_prediction, tf.int32), 1)
      return tf.reduce_sum(tf.cast(correct_batch, tf.float32)), correct_batch
       
def create_generator(generator_inputs, generator_outputs_channels, name="generator", reuse=False):
#    U_Net结构
    with tf.variable_scope(name,reuse=reuse):
        layers = []
        with tf.variable_scope("encoder_1"):
            # encoder_1: [batch, 64, 64, 1 ] => [batch, 32, 32, 32]
            convolved = conv(generator_inputs, 32, 3, pad='SAME')
            convolved = tf.nn.relu(convolved)
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
                convolved = conv(output, out_channels, 3, pad='SAME')
                convolved = tf.nn.relu(convolved)
                output = tf.nn.max_pool(convolved,[1,2,2,1],[1,2,2,1],padding='SAME') 
                layers.append(output)
    
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
                output = deconv(inputs, out_channels, 3, pad='SAME')
                output = tf.nn.relu(output)
    
                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
    
                layers.append(output)
    
        # decoder_4: [batch, 32, 32, ngf ] => [batch, 64, 64, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            inputs = tf.concat([layers[-1], layers[0]], axis=3)
            output = deconv(inputs, generator_outputs_channels,3,pad='SAME')
            output = tf.tanh(output)
            layers.append(output)
        return layers[-1]


# 判别器网络
def create_discriminator(inputs_o, inputs_r, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        inputs = tf.concat([inputs_o, inputs_r], axis=3)  #  64*64*2
        layer1 = conv(inputs, 32, 3, pad='SAME')    # 64*64*2 --> 64*64*32
        layer1 = tf.nn.relu(layer1)
        pool1 = tf.nn.max_pool(layer1,[1,2,2,1],[1,2,2,1],padding='SAME') 

        layer2 = conv(pool1, 64, 3, pad='SAME')   # 32*32*32 --> 32*32*64
        layer2 = tf.nn.relu(layer2)
        pool2 = tf.nn.max_pool(layer2,[1,2,2,1],[1,2,2,1],padding='SAME') 

        layer3 = conv(pool2, 128, 3, pad='SAME')   # 16*16*64 --> 16*16*128
        layer3 = tf.nn.relu(layer3)
        pool3 = tf.nn.max_pool(layer3,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer4 = conv(pool3, 256, 3, pad='SAME')   # 8*8*128 --> 8*8*256
        layer4 = tf.nn.relu(layer4)
        pool4 = tf.nn.max_pool(layer4,[1,2,2,1],[1,2,2,1],padding='SAME')   # 4*4*256
        
        layer5 = conv(pool4, 512, 3, pad='SAME')   # 4*4*256 --> 4*4*512
        layer5 = tf.nn.relu(layer5)
        pool5 = tf.nn.max_pool(layer5,[1,2,2,1],[1,2,2,1],padding='SAME')   # 2*2*512
        
        batch_size = int(pool5.get_shape()[0])
        dense = tf.reshape(pool5, [batch_size,-1])
        dense1 = tf.layers.dense(inputs=dense, units=1024)
        dense1 = tf.nn.relu(dense1)
        dense2 = tf.layers.dense(inputs=dense1, units=256)
        dense2 = tf.nn.relu(dense2)
        dense3 = tf.layers.dense(inputs=dense2, units=1)
        output = tf.sigmoid(dense3)
        return output

        
def gd_model_o2r(inputs_o, inputs_r):
    # opt --> nir
#    with tf.variable_scope("sar2opt"):
        out_channels = int(inputs_r.get_shape()[-1])
        g_outputs = create_generator(inputs_o, out_channels)  # fake_optical
        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            # 2x [batch, height, width, channels] => [batch, 1]
            predict_real= create_discriminator(inputs_o, inputs_r)
        with tf.name_scope("fake_discriminator"):
            # 2x [batch, height, width, channels] => [batch, 1]
            predict_fake = create_discriminator(inputs_o, g_outputs, reuse=True)
    
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
            gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight   # 为什么权重相差这么大
            
        return  gen_loss, discrim_loss, g_outputs, [gen_loss_GAN, gen_loss_L1]     
        




