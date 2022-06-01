# -*- coding: utf-8 -*-
"""
Created on 2018.11.9 11:03:33 

@author: Administrator

双分支 + 特征map + 交叉熵损失函数 + 参数共享 + 差值求绝对值
改进：将上下两个支路的特征向量结合起来，得到最终判别结果

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
#        4*4*256
        return [layer1, pool1, pool2, pool3, pool4], pool4

'''
matching layers
update: 2018.11.9

'''
    
def matching(feature_map_1,feature_map_2): 
    with tf.variable_scope('match_layers_1'):
#        all_feature = tf.concat([feature_map_1,feature_map_2], axis=3)  # 两个特征连接 4*4*512
        diff_feature = tf.abs(feature_map_1 - feature_map_2)
        layer5 = conv(diff_feature, 512, 3, pad='SAME')    
        layer5 = tf.nn.relu(layer5)
        pool5 = tf.nn.max_pool(layer5,[1,2,2,1],[1,2,2,1],padding='SAME') # 2*2*512
    
        batch_size = int(pool5.get_shape()[0])
        dense = tf.reshape(pool5, [batch_size,-1])  # reshape as a tensor 2048
        dense1 = tf.layers.dense(dense, 1024)
        dense1 = tf.nn.relu(dense1)
        
    with tf.variable_scope('match_layers_2'):    
        dense2 = tf.layers.dense(dense1, 256)
        dense2 = tf.nn.relu(dense2)
        dense3 = tf.layers.dense(dense2, 1)
        output = tf.sigmoid(dense3)   
        return output, dense1, [pool5, dense1, dense2]

'''
add the resnet 
update: 2018.11.10

'''
def matching_res(layer1_res, pool1_res, pool2_res, pool3_res, pool4_res): 
    with tf.variable_scope('match_res_1'):
        layer1 = conv(layer1_res, 32, 1, pad='SAME') 
        layer1 = tf.nn.relu(layer1)
        pool1 = tf.nn.max_pool(layer1,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        pool1_con = tf.concat([pool1, pool1_res], axis=3)
        layer2 = conv(pool1_con, 64, 1, pad='SAME') 
        layer2 = tf.nn.relu(layer2)
        pool2 = tf.nn.max_pool(layer2,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        pool2_con = tf.concat([pool2, pool2_res], axis=3)
        layer3 = conv(pool2_con, 128, 1, pad='SAME')   
        layer3 = tf.nn.relu(layer3)
        pool3 = tf.nn.max_pool(layer3,[1,2,2,1],[1,2,2,1],padding='SAME')
        
        pool3_con = tf.concat([pool3, pool3_res], axis=3)
        layer4 = conv(pool3_con, 256, 1, pad='SAME')  
        layer4 = tf.nn.relu(layer4)
        pool4 = tf.nn.max_pool(layer4,[1,2,2,1],[1,2,2,1],padding='SAME')
        
        pool4_con = tf.concat([pool4, pool4_res], axis=3)
        layer5 = conv(pool4_con, 512, 1, pad='SAME')  
        layer5 = tf.nn.relu(layer5)
        pool5 = tf.nn.max_pool(layer5,[1,2,2,1],[1,2,2,1],padding='SAME')  # 2*2*512
        
        # 2*2*512=2048
        batch_size1 = int(pool5.get_shape()[0])
        dense = tf.reshape(pool5, [batch_size1,-1])

        dense1 = tf.layers.dense(dense, 1024)
        dense1 = tf.nn.relu(dense1)
        
    with tf.variable_scope('match_res_2'):
        dense2 = tf.layers.dense(dense1, 256)
        dense2 = tf.nn.relu(dense2)
        
        dense3 = tf.layers.dense(dense2, 1)
        output = tf.sigmoid(dense3)   
        return output, dense1, [pool1, pool2, pool3, pool4, pool5, dense1, dense2]

'''
'''
def matching_res_pro(dense1, dense1_res):
    dense_con = tf.concat([dense1, dense1_res], axis=1) # 1024 + 1024 = 2048
    dense1 = tf.layers.dense(dense_con, 1024)
    dense1 = tf.nn.relu(dense1)
    dense2 = tf.layers.dense(dense1, 256)
    dense2 = tf.nn.relu(dense2)
    dense3 = tf.layers.dense(dense2, 1)
    output = tf.sigmoid(dense3) 
    return output, [dense1, dense2, dense3]

'''
network model
'''
def features_matching(inputs_p1, inputs_p2):
    with tf.variable_scope("matching"):
        layers_1,feature_map_1 = features(inputs_p1, 'feature_layers') # 4*4*256
        layers_2,feature_map_2 = features(inputs_p2, 'feature_layers',reuse=True) # 4*4*256
        output, dense1, feature_maps = matching(feature_map_1,feature_map_2)   
        return output, dense1, [layers_1,layers_2,feature_maps]
      
'''
features_matching_res network
update: 2018.11.10

'''
def features_matching_res(inputs_p1, inputs_p2):
    with tf.variable_scope("matching"):
        layers_1,feature_map_1 = features(inputs_p1, 'feature_layers') # 4*4*256
        layers_2,feature_map_2 = features(inputs_p2, 'feature_layers',reuse=True) # 4*4*256
        output, dense1, feature_maps = matching(feature_map_1,feature_map_2)    
    with tf.variable_scope('match_res'):
        output_res, dense1_res, res_feature_maps = matching_res(abs(layers_1[0]-layers_2[0]), abs(layers_1[1]-layers_2[1]), abs(layers_1[2]-layers_2[2]), 
                                                    abs(layers_1[3]-layers_2[3]), abs(layers_1[4]-layers_2[4]))
    with tf.variable_scope('match_pro'):
        output_pro, pro_feature_maps = matching_res_pro(dense1, dense1_res)
    
        return output, output_res, output_pro, [layers_1, layers_2, feature_maps, res_feature_maps, pro_feature_maps]  


'''
cross loss function
'''
def match_network(inputs_p1, inputs_p2, label_m):

    match_output, all_features = features_matching(inputs_p1, inputs_p2)    
    match_loss =  tf.reduce_mean(-(label_m * tf.log(match_output + EPS) + (1 - label_m) * tf.log(1 - match_output + EPS)))    
    return match_loss, match_output, all_features

def match_network_res(inputs_p1, inputs_p2, label_m):

    match_output, match_output_res, match_output_pro, all_features = features_matching_res(inputs_p1, inputs_p2)    
    match_loss =  tf.reduce_mean(-(label_m * tf.log(match_output + EPS) + (1 - label_m) * tf.log(1 - match_output + EPS)))    
    match_loss_res =  tf.reduce_mean(-(label_m * tf.log(match_output_res + EPS) + (1 - label_m) * tf.log(1 - match_output_res + EPS)))  
    match_loss_pro =  tf.reduce_mean(-(label_m * tf.log(match_output_pro + EPS) + (1 - label_m) * tf.log(1 - match_output_pro + EPS)))  
    
    return match_loss, match_loss_res, match_loss_pro, match_output, match_output_res, match_output_pro, all_features




