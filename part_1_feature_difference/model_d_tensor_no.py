# -*- coding: utf-8 -*-
"""
Created on 2018.11.9 11:03:33 

@author: Administrator

双分支 + 特征向量 + 交叉熵损失函数 + 参数fei共享

"""

import tensorflow as tf
import numpy as np

EPS = 1e-12

def conv(batch_input, out_channels, filter_size, pad, stride=1):
    conv = tf.layers.conv2d(batch_input, out_channels,filter_size,strides=(stride,stride), padding=pad)
    return conv

def batchnorm(inputs):
    normalized = tf.layers.batch_normalization(inputs, axis=-1)
    return normalized

def evaluation(logits, labels):
      correct_prediction = tf.equal(logits, labels)
      correct_batch = tf.reduce_mean(tf.cast(correct_prediction, tf.int32), 1)
      return tf.reduce_sum(tf.cast(correct_batch, tf.float32)), correct_batch

'''
# image patch features abstract layers
update: 2018.11.9
5*conv + fc

'''
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
#        
#        layer5 = conv(pool4, 256, 3, pad='SAME')              # 4*4*256      
#        layer5 = tf.nn.relu(layer5)
        pool5 = tf.nn.max_pool(pool4,[1,2,2,1],[1,2,2,1],padding='SAME')

#        2*2*256=1024
        batch_size = int(pool5.get_shape()[0])
        dense = tf.reshape(pool5, [batch_size,-1])
        
        return [pool1, pool2, pool3, pool4, pool5, dense], dense

'''
matching layers
update: 2018.11.9

'''
def matching(dense_1,dense_2): 
    with tf.variable_scope('match_layers'):
        all_feature = tf.concat([dense_1, dense_2], axis=1)  # 两个特征连接 1024*2=2048
        dense1 = tf.layers.dense(all_feature, 1024)
        dense1 = tf.nn.relu(dense1)
        
        dense2 = tf.layers.dense(dense1, 256)
        dense2 = tf.nn.relu(dense2)
         
        dense3 = tf.layers.dense(dense2, 1)
        output = tf.sigmoid(dense3)   
        return output, [dense1,dense2]

'''
network model
'''
def features_matching(inputs_p1, inputs_p2):
    with tf.variable_scope("matching"):
        layers_1,dense_1 = features(inputs_p1, 'feature_layers_1') # 4*4*256
        layers_2,dense_2 = features(inputs_p2, 'feature_layers_2') # 4*4*256
        output,denses = matching(dense_1,dense_2)   
        return output ,[layers_1,layers_2,denses]

'''
cross loss function
'''

def match_network(inputs_p1, inputs_p2, label_m):

    match_output, all_features = features_matching(inputs_p1, inputs_p2)    
    match_loss =  tf.reduce_mean(-(label_m * tf.log(match_output + EPS) + (1 - label_m) * tf.log(1 - match_output + EPS)))    
    return match_loss, match_output, all_features






