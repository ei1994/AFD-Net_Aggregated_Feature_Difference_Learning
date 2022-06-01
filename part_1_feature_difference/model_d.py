# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:03:33 2018

@author: Administrator
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

def features(img, name, reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        layer1 = conv(img, 32, 3, pad='SAME') 
        layer1 = tf.nn.relu(layer1)
        pool1 = tf.nn.max_pool(layer1,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer2 = conv(pool1, 64, 3, pad='SAME')  
        layer2 = tf.nn.relu(layer2)
        pool2 = tf.nn.max_pool(layer2,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer3 = conv(pool2, 128, 3, pad='SAME')    
        layer3 = tf.nn.relu(layer3)
        pool3 = tf.nn.max_pool(layer3,[1,2,2,1],[1,2,2,1],padding='SAME')
        
        layer4 = conv(pool3, 256, 3, pad='SAME')    
        layer4 = tf.nn.relu(layer4)
        pool4 = tf.nn.max_pool(layer4,[1,2,2,1],[1,2,2,1],padding='SAME')
        # 4*4*256=4096
#        batch_size1 = int(pool4.get_shape()[0])
#        dense = tf.reshape(pool4, [batch_size1,-1])
        
        return [layer1, pool1, pool2, pool3, pool4], pool4

# 4conv + 3fc
def matching(f1,f2,keep_prob): 
    with tf.variable_scope('match_layers'):
        all_feature = tf.concat([f1, f2], axis=3)  # 两个特征连接 4*4*512
        layer5 = conv(all_feature, 512, 3, pad='SAME')    
        layer5 = tf.nn.relu(layer5)
        pool5 = tf.nn.max_pool(layer5,[1,2,2,1],[1,2,2,1],padding='SAME')
        
        batch_size1 = int(pool5.get_shape()[0])
        dense = tf.reshape(pool5, [batch_size1,-1])
        dense1 = tf.layers.dense(dense, 1024)
        dense1 = tf.nn.relu(dense1)
        # dropout layers
        dense1 = tf.nn.dropout(dense1, keep_prob)
        
        dense2 = tf.layers.dense(dense1, 256)
        dense2 = tf.nn.relu(dense2)
        dense2 = tf.nn.dropout(dense2, keep_prob)
         
        dense3 = tf.layers.dense(dense2, 1)
        output = tf.sigmoid(dense3)   
        return output, [pool5,dense1,dense2]

# 双分支特征向量, tensor = 3conv + 2fc
def features_matching(inputs_p1, inputs_p2,keep_prob):
    with tf.variable_scope("matching"):
        layers_1,f1 = features(inputs_p1, 'inputs_p') # 4*4*256
        layers_2,f2 = features(inputs_p2, 'inputs_p',reuse=True) # 4*4*256
        output,denses = matching(f1,f2,keep_prob)   
        return output ,[layers_1,f1,layers_2,f2,denses]

# 3conv + 5fc
def matching_res(layer1_res, pool1_res, pool2_res, pool3_res, pool4_res,keep_prob): 
   
        layer1 = conv(layer1_res, 32, 1, pad='SAME') 
#        layer1 = batchnorm(layer1)
        layer1 = tf.nn.relu(layer1)
        pool1 = tf.nn.max_pool(layer1,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        pool1_con = tf.concat([pool1, pool1_res], axis=3)
        layer2 = conv(pool1_con, 64, 1, pad='SAME') 
#        layer2 = batchnorm(layer2)
        layer2 = tf.nn.relu(layer2)
        pool2 = tf.nn.max_pool(layer2,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        pool2_con = tf.concat([pool2, pool2_res], axis=3)
        layer3 = conv(pool2_con, 128, 1, pad='SAME')   
#        layer3 = batchnorm(layer3)
        layer3 = tf.nn.relu(layer3)
        pool3 = tf.nn.max_pool(layer3,[1,2,2,1],[1,2,2,1],padding='SAME')
        
        pool3_con = tf.concat([pool3, pool3_res], axis=3)
        layer4 = conv(pool3_con, 256, 1, pad='SAME')  
#        layer3 = batchnorm(layer3)
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
        # dropout layers
        dense1 = tf.nn.dropout(dense1, keep_prob)
        
        dense2 = tf.layers.dense(dense1, 256)
        dense2 = tf.nn.relu(dense2)
        dense2 = tf.nn.dropout(dense2, keep_prob)
        
        dense3 = tf.layers.dense(dense2, 1)
        output = tf.sigmoid(dense3)   
        return output,[pool1, pool2, pool3, pool4, pool5, dense1,dense2]
    
# 双分支特征向量,参差损失, tensor = 3conv + 2fc
def features_matching_res(inputs_p1, inputs_p2, keep_prob):
    with tf.variable_scope("matching"):
        layers_1,f1 = features(inputs_p1, 'inputs_p') # 1024
        layers_2,f2 = features(inputs_p2, 'inputs_p',reuse=True) # 1024
        output ,denses = matching(f1,f2,keep_prob)  
    with tf.variable_scope('match_res'):
        output_res, res_features = matching_res(abs(layers_1[0]-layers_2[0]), abs(layers_1[1]-layers_2[1]), abs(layers_1[2]-layers_2[2]), 
                                                    abs(layers_1[3]-layers_2[3]), abs(layers_1[4]-layers_2[4]),keep_prob)
        return output, output_res, [layers_1, f1, layers_2, f2, denses, res_features]    
    
# 双分支特征向量,共享参数
def matchLoss(inputs_p1, inputs_p2, label_m, keep_prob):
    #    匹配损失
    m_output, all_features = features_matching(inputs_p1, inputs_p2,keep_prob)    
    match_loss =  tf.reduce_mean(-(label_m * tf.log(m_output + EPS) + (1 - label_m) * tf.log(1 - m_output + EPS)))    
    return match_loss, m_output, all_features

# 双分支特征向量,共享参数
def matchLoss_res(inputs_p1, inputs_p2, label_m, keep_prob):
    #    匹配损失
    m_output, m_output_res, all_features = features_matching_res(inputs_p1, inputs_p2, keep_prob)    
    match_loss =  tf.reduce_mean(-(label_m * tf.log(m_output + EPS) + (1 - label_m) * tf.log(1 - m_output + EPS)))    
    match_loss_res =  tf.reduce_mean(-(label_m * tf.log(m_output_res + EPS) + (1 - label_m) * tf.log(1 - m_output_res + EPS)))  
    return match_loss,match_loss_res,  m_output, m_output_res, all_features





    
def join_model(inputs_p1, inputs_p2, label_m, inputs_p, label_c, points_num):        
#    匹配损失
        m_output, f1, f2 = features_matching(inputs_p1, inputs_p2)    
        match_loss =  tf.reduce_mean(-(label_m * tf.log(m_output + EPS) + (1 - label_m) * tf.log(1 - m_output + EPS)))

#    分类损失
        c_output, f = classifier(inputs_p, points_num)
        
        class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_c,logits=c_output))
        
        return match_loss, class_loss, m_output







