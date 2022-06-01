
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:56:02 2018

@author: DELL
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
import scipy.misc as misc
import numpy as np


SAVA_PATH = 'mapping_data'
TRAIN_FILE = 'map_country.tfrecord'
BATCH_SIZE = 10
img_size = 64



def read_and_decode_train(filename_queue):  
    # create a reader from file queue  
    reader = tf.TFRecordReader()  
    #reader从 TFRecord 读取内容并保存到 serialized_example 中 
    _, serialized_example = reader.read(filename_queue)  

    features = tf.parse_single_example(     # 读取 serialized_example 的格式 
        serialized_example,  
        features={  
            'label_raw': tf.FixedLenFeature([], tf.string),  
            'image_raw': tf.FixedLenFeature([], tf.string)      
        }  
    )  # 解析从 serialized_example 读取到的内容  
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [img_size, img_size*2])
#    img_gray = tf.image.rgb_to_grayscale(img)
    img = tf.cast(img, tf.float32) * (1. / 255.0)      #[0,1]
    label = tf.decode_raw(features['label_raw'], tf.int32)
    label = tf.reshape(tf.cast(label, tf.float32),shape=(1,))
    return img,label
#   return tf.cast(resize_image, tf.float32), tf.cast(reshape_label, tf.float32)

def read_and_decode_test(filename_queue):  
    # create a reader from file queue  
    reader = tf.TFRecordReader()  
    #reader从 TFRecord 读取内容并保存到 serialized_example 中 
    _, serialized_example = reader.read(filename_queue)  

    features = tf.parse_single_example(     # 读取 serialized_example 的格式 
        serialized_example,  
        features={  
            'label_raw': tf.FixedLenFeature([], tf.string),  
            'image_raw': tf.FixedLenFeature([], tf.string)      
        }  
    )  # 解析从 serialized_example 读取到的内容  
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [img_size, img_size*2])
#    img_gray = tf.image.rgb_to_grayscale(img)
    img = tf.cast(img, tf.float32) * (1. / 255.0)      #[0,1]
    label = tf.decode_raw(features['label_raw'], tf.int32)
    label = tf.reshape(tf.cast(label, tf.float32),shape=(1,))
    return img,label

def batch_inputs(filename_queue, train, batch_size):
      #创建文件队列,不限读取的数量  
  with tf.name_scope('input'):
    if train:
        image, label = read_and_decode_train(filename_queue)
        images, labels = tf.train.shuffle_batch([image, label],
                                                       batch_size=batch_size,
                                                       num_threads=6,
                                                       capacity=2000 + 3 * batch_size,
                                                       min_after_dequeue=2000)
    else:
        image, label = read_and_decode_test(filename_queue)
        images, labels = tf.train.batch([image, label],
                                               batch_size=batch_size,
                                               num_threads=6,
                                               capacity=2000 + 3 * batch_size)
    return images, labels

if __name__ == '__main__':
    
    init = tf.global_variables_initializer()
#    filename = os.path.join(SAVA_PATH, TRAIN_FILE)
    filename = '/home/ws/文档/wrj/mapping_data_all/mapping_data/map_country.tfrecord'
    filename_queue = tf.train.string_input_producer([filename],num_epochs=10 , shuffle=True)
    img_batch, label_batch = batch_inputs(filename_queue, train = True, batch_size = BATCH_SIZE)
    with tf.Session() as sess:
#        sess.run(init)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程
        #启动QueueRunner, 此时文件名队列已经进队。
        threads=tf.train.start_queue_runners(sess=sess,coord=coord) 
        try:
            for i in range(100):
                img, label = sess.run([img_batch, label_batch]) 
                for j in range(img.shape[0]):
                    misc.imsave(str(j)+'_read.png', np.squeeze(img[j]))
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)
  
  
