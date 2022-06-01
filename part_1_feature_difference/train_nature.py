#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:37:06 2017

@author: wrj
"""
'''
双分支参数共享 + 分类损失
参差损失
'''

import numpy as np
import math
import time
import model_d as model
import read_data
import tensorflow as tf
import os
from datetime import datetime
import logging
from sklearn.metrics import roc_curve, auc

BATCH_SIZE_matching=100
BATCH_SIZE_iden=200
epoch = 41
learning_rate = 2e-4
image_width = 64
image_height = 64
KEEP_prob = 1.0

checkpoint_dir = 'ckpt_country'
checkpoint_dir_g = 'ckpt_country'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
checkpoint_file_g = os.path.join(checkpoint_dir_g, 'model.ckpt')
train_dir='summary_country'

def initLogging(logFilename='record_country.log'):
  """Init for logging
  """
  logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets, dtype=np.int32).reshape(-1)]    
   
def norm(patch_1):
    np.seterr(divide='ignore', invalid='ignore')
    patch_1 = np.reshape(patch_1,(patch_1.shape[0],-1))
#    patch_max = np.max(patch_1, (1))
#    patch_max = patch_max[:,np.newaxis]
#    patch_min = np.min(patch_1, (1))
#    patch_min = patch_min[:,np.newaxis]
#    patch_dif = patch_max - patch_min
#    patch_1 = (patch_1 - patch_min)/patch_dif
    patch_1 = patch_1 /255.0
    patch_1 = np.reshape(patch_1,(patch_1.shape[0],64,64))
    patch_1 = patch_1[:,:,:,np.newaxis]
    return patch_1

def mc_train():
    initLogging()
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    checkpoints_dir = 'checkpoints/{}'.format(current_time)
    try:
        os.makedirs(checkpoint_dir)
        os.makedirs(checkpoints_dir)
    except os.error:
        pass
    
    data1 = np.load('data_all/country.npz')
    train_matching_y = data1['arr_1'][:,np.newaxis]
    numbers_train = train_matching_y.shape[0]  #训练集总数
    epoch_steps = np.int(numbers_train / BATCH_SIZE_matching)  +1  # 一个epoch有多少个steps

    all_loss = np.array([])
    graph = tf.Graph()
    with graph.as_default():
        inputs_p1 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width,1], name='inputs_p1')
        inputs_p2 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width,1], name='inputs_p2')
        label_m = tf.placeholder(tf.float32, [BATCH_SIZE_matching, 1], name='label_m')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # 训练 M
        match_loss, m_output, all_features = model.matchLoss(inputs_p1, inputs_p2, label_m,keep_prob)
        match_out = tf.round(m_output)
        m_correct,m_numbers = model.evaluation(match_out, label_m)
        
        m_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(match_loss)      
        
        filename = 'data_all/country/country.tfrecord'
        filename_queue = tf.train.string_input_producer([filename],num_epochs=epoch,shuffle=True)
        img_batch, label_batch = read_data.batch_inputs(filename_queue,  train = True, batch_size = BATCH_SIZE_matching)
        tf.summary.scalar('mathing_loss', match_loss)

        summary = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=20)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            sess.run(tf.local_variables_initializer())
            sess.run(init)
            
            try:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                step = 0
               
                while not coord.should_stop():
                        start_time = time.time()
                        step = step + 1
                        batch, l_batch = sess.run([img_batch, label_batch])
                        l_batch = l_batch
                        x_batch = batch[:,:,:64,np.newaxis]
                        y_batch = batch[:,:,64:,np.newaxis]
                        feed_dict = {inputs_p1:x_batch, inputs_p2:y_batch, label_m:l_batch, keep_prob:KEEP_prob }
                        _, m_loss, m_output_= sess.run([m_train_opt,match_loss,m_output], feed_dict = feed_dict)
                        
                        if step % 10 ==0:
                            loss_write = np.array([[step, m_loss]])
                            if step ==10:
                                all_loss = loss_write
                            else:
                                all_loss = np.concatenate((all_loss, loss_write))
                        
                        if step % 100 == 0:
                            duration = time.time() - start_time
                            summary_str = sess.run(summary, feed_dict=feed_dict)
                            summary_writer.add_summary(summary_str, step)
                            summary_writer.flush()
                            
                            logging.info('>> Step %d run_train: matching_loss = %.3f  (%.3f sec)'
                                          % (step, m_loss, duration))
                            
                        if (step % epoch_steps == 0) and ((step / epoch_steps)%5 == 0):
                            logging.info('>> %s Saving in %s' % (datetime.now(), checkpoint_dir))
                            saver.save(sess, checkpoint_file, global_step=step)
                       
            except KeyboardInterrupt:
                print('INTERRUPTED')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)

            finally:
                saver.save(sess, checkpoint_file, global_step=step)
                np.save(os.path.join(checkpoint_dir,'ckpt_country'), all_loss)
                print('Model saved in file :%s'%checkpoint_dir)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)
            
#'''
def gfm_test(file_name):
    data1 = np.load('data_all/'+file_name+'.npz')
    train_matching_y = data1['arr_1'][:,np.newaxis]
    numbers_train = train_matching_y.shape[0]  #训练集总数
#    epoch_steps = np.int(numbers_train / BATCH_SIZE_matching)  +1  # 一个epoch有多少个steps
    graph = tf.Graph()
    with graph.as_default():
        inputs_p1 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width, 1], name='inputs_p1')
        inputs_p2 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width, 1], name='inputs_p2')
        label_m = tf.placeholder(tf.float32, [BATCH_SIZE_matching, 1], name='label_m')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # 训练 M
        match_loss, m_output, all_features = model.matchLoss(inputs_p1, inputs_p2, label_m,keep_prob,)
        match_out = tf.round(m_output)
        m_correct,m_numbers = model.evaluation(match_out, label_m)
        
        filename = 'data_all/country/'+file_name+'.tfrecord'
        filename_queue = tf.train.string_input_producer([filename],num_epochs=1,shuffle=False)
        img_batch, label_batch = read_data.batch_inputs(filename_queue,  train = False, batch_size = BATCH_SIZE_matching)
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('ckpt_country'))
#            saver.restore(sess, 'ckpt_mc_ottawa/model.ckpt-4000')
            m_count = 0  # Counts the number of correct predictions.
            num = numbers_train
            matching_out = np.array([])
            matching_label = np.array([])
            
            try:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                step_test = 0
                try:
                    while not coord.should_stop():
                        step_test = step_test + 1
                        batch, l_batch = sess.run([img_batch, label_batch])
                        l_batch = l_batch
                        x_batch = batch[:,:,:64,np.newaxis]
                        y_batch = batch[:,:,64:,np.newaxis]
                        feed_dict = {inputs_p1:x_batch, inputs_p2:y_batch, label_m:l_batch,keep_prob:KEEP_prob}
                        m_correct_, m_output_ = sess.run([m_correct,m_output], feed_dict = feed_dict)
                            
                        if step_test == 1:
                            matching_out = m_output_
                            matching_label = l_batch
                        elif(l_batch.size == BATCH_SIZE_matching):
                                matching_out = np.concatenate((matching_out, m_output_))
                                matching_label = np.concatenate((matching_label, l_batch))
                                
                        m_count = m_count + m_correct_.astype(int)
                        if step_test % 100 == 0:
                            print('Step %d run_test: batch_precision = %.2f  '% (step_test, m_correct_/BATCH_SIZE_matching))
                except Exception as e:
                    coord.request_stop(e)
                m_precision = float(m_count) / num

                print('  Num examples: %d  Num correct: %d  match Precision : %0.04f  ' %(num, m_count, m_precision))
                save_file = open('test_nature.txt','a')
                save_file.write(file_name + '\n'+'match Precision : '+str(m_precision))
                save_file.write('\n')
                
        #            绘制ROC曲线
                fpr,tpr,threshold = roc_curve(matching_label, matching_out) ###计算真正率和假正率  
                roc_auc = auc(fpr,tpr) ###计算auc的值 
                save_file.write('match roc_auc : '+str(roc_auc))
                save_file.write('\n')
                save_file.write('\n')
                save_file.close()
                try:
                    os.makedirs('plot_curve/roc/nature')
                except os.error:
                    pass
                np.savez('plot_curve/roc/nature/match_'+file_name, fpr,tpr)
            except KeyboardInterrupt:
                print('INTERRUPTED')
                coord.request_stop()
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)

if __name__ == '__main__':
    
#    mc_train()
    
    test_data = ['field', 'forest','indoor','mountain','oldbuilding','street','urban','water']
    for i in range(8):
        gfm_test(test_data[i])
        
        
        