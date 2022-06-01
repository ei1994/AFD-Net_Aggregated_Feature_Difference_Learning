#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2018.11.9

@author: wrj
"""
'''
双分支 + 特征map32 + 交叉熵损失函数 + 参数共享
使用生成器进行图像的转换。
'''

import numpy as np
import math
import time
import model_d_map32_diff as model
import model_gd_diff_pro as model_g
import read_data as read
import tensorflow as tf
import os
from datetime import datetime
import logging
from sklearn.metrics import roc_curve, auc

os.environ['CUDA_VISIBLE_DEVICES']='0'

BATCH_SIZE_matching=100
epoch = 41
learning_rate = 2e-4
image_width = 64
image_height = 64

checkpoint_dir = 'ckpt_map_gd_pro'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
checkpoint_dir_g = 'ckpt_g_g2r_pro'
checkpoint_file_g = os.path.join(checkpoint_dir_g+'_country', 'model.ckpt')
train_dir='summary'

def initLogging(logFilename='record_map_gd_pro.log'):
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


def mc_train():
    initLogging()
    current_time = datetime.now().strftime('%Y-%m-%d')
    try:
        os.makedirs(checkpoint_dir)
    except os.error:
        pass
#    
    data1 = np.load('/home/ws/文档/wrj/data_all_my/country.npz')
    train_matching_y = data1['arr_1'][:,np.newaxis]
    numbers_train = train_matching_y.shape[0]  #训练集总数
    epoch_steps = np.int(numbers_train / BATCH_SIZE_matching)  +1  # 一个epoch有多少个steps

    all_loss = np.array([])
    graph = tf.Graph()
    with graph.as_default():
        inputs_p1 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width,1], name='inputs_gray')
        inputs_p2 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width,1], name='inputs_nir')
        label_m = tf.placeholder(tf.float32, [BATCH_SIZE_matching, 1], name='label_m')

        inputs_p1_ = model_g.preprocess(inputs_p1)
        g_outputs = model_g.create_generator(inputs_p1_, 1)
        g_outputs_ = model_g.deprocess(g_outputs)
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        # 训练 M
        match_loss, match_output, all_features = model.match_network( g_outputs_, inputs_p2, label_m)
        match_out = tf.round(match_output)
        m_correct,m_numbers = model.evaluation(match_out, label_m)
        
        match_tvars = [var for var in tf.trainable_variables() if var.name.startswith("matching")]
        m_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(match_loss, var_list=match_tvars)      
        
        filename = '/home/ws/文档/wrj/data_all/country/country.tfrecord'
        filename_queue = tf.train.string_input_producer([filename],num_epochs=epoch,shuffle=True)
        img_batch, label_batch = read.batch_inputs(filename_queue,  train = True, batch_size = BATCH_SIZE_matching)
        tf.summary.scalar('mathing_loss', match_loss)

        summary = tf.summary.merge_all()
        saver_g = tf.train.Saver(var_list=gen_tvars)
        saver = tf.train.Saver(max_to_keep=20)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            sess.run(tf.local_variables_initializer())
            sess.run(init)
            saver_g.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_g+'_all'))
            
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
                        feed_dict = {inputs_p1:x_batch, inputs_p2:y_batch, label_m:l_batch}
                        _, m_loss, m_output_= sess.run([m_train_opt,match_loss,match_output], feed_dict = feed_dict)
                        
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
                            
                        if (step % epoch_steps == 0) and ((step / epoch_steps)%3 == 0):
                            current_epoch = int(step / epoch_steps)
                            
                            if current_epoch > 25:
                              logging.info('>> %s Saving in %s' % (datetime.now(), checkpoint_dir))
                              saver.save(sess, checkpoint_file, global_step=current_epoch)
                              mc_test_all(current_epoch)
                       
            except KeyboardInterrupt:
                print('INTERRUPTED')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)

            finally:
                saver.save(sess, checkpoint_file, global_step=step)
                np.save(os.path.join(checkpoint_dir,'ckpt_map_gd_pro'), all_loss)
                print('Model saved in file :%s'%checkpoint_dir)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)
            
'''
test the network
'''
def mc_test(file_name, number):
    data = np.load('/home/ws/文档/wrj/data_all_test/test_'+file_name+'.npz')
#    data = np.load('/home/ws/文档/wrj/data_all/'+file_name+'.npz')
    train_matching_y = data['arr_1'][:,np.newaxis]
    numbers_train = train_matching_y.shape[0]  #训练集总数
    
    graph = tf.Graph()
    with graph.as_default():
        inputs_p1 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width, 1], name='inputs_gray')
        inputs_p2 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width, 1], name='inputs_nir')
        label_m = tf.placeholder(tf.float32, [BATCH_SIZE_matching, 1], name='label_m')

        inputs_p1_ = model_g.preprocess(inputs_p1)
        g_outputs = model_g.create_generator(inputs_p1_, 1)
        g_outputs_ = model_g.deprocess(g_outputs)
        # 训练 M
        match_loss, match_output, all_features = model.match_network(  g_outputs_,inputs_p2, label_m)
        match_out = tf.round(match_output)
        m_correct,m_numbers = model.evaluation(match_out, label_m)
        
#        filename = '/home/ws/文档/wrj/data_all/country/'+file_name+'.tfrecord'
        filename = '/home/ws/文档/wrj/data_all_test/test_data/test_'+file_name+'.tfrecord'
        filename_queue = tf.train.string_input_producer([filename],num_epochs=1,shuffle=False)
        img_batch, label_batch = read.batch_inputs(filename_queue,  train = False, batch_size = BATCH_SIZE_matching)
        
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        saver_g = tf.train.Saver(var_list=gen_tvars)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, 'ckpt_map_gd_pro/model.ckpt-27')
#            saver.restore(sess, tf.train.latest_checkpoint('ckpt_map_gd_pro'))
            saver_g.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_g+'_all'))
#            saver.restore(sess, 'ckpt_tensor/model.ckpt-4000')
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
                        feed_dict = {inputs_p1:x_batch, inputs_p2:y_batch, label_m:l_batch}
                        m_correct_, m_output_ = sess.run([m_correct,match_output], feed_dict = feed_dict)
                            
                        if step_test == 1:
                            matching_out = m_output_
                            matching_label = l_batch
                        elif(l_batch.size == BATCH_SIZE_matching):
                                matching_out = np.concatenate((matching_out, m_output_))
                                matching_label = np.concatenate((matching_label, l_batch))
                                
                        m_count = m_count + m_correct_.astype(int)
                        if step_test % 100 == 0:
                            print('Step %d/%d  run_test: batch_precision = %.2f  '% (step_test, num/100, m_correct_/BATCH_SIZE_matching))
                except Exception as e:
                    coord.request_stop(e)
                m_precision = float(m_count) / num

                print('Num examples: %d  Num correct: %d  match Precision : %0.04f  ' %(num, m_count, m_precision))
                save_file = open('test_map_gd_pro.txt','a')
                save_file.write(file_name + '    epoch: ' + str(number) +'\n'+'num correct: '+str(m_count)+'/'+str(num)+'     match precision : '+str(m_precision))
                save_file.write('    ')
                
#                 计算 tp,tn,fp,fn 
                pre_out = np.round(matching_out)
                pre_out = pre_out.astype(int)
                lab_out = matching_label.astype(int)
                tp = np.sum(pre_out & lab_out)
                tn = m_count - tp
                fp = np.sum(pre_out) - tp
                fn = np.sum(lab_out) - tp
                save_file.write(file_name + '    tp: ' + str(tp)+ '    tn: ' + str(tn) \
                                            + '    fp: ' + str(fp)+ '    fn: ' + str(fn))
        #       绘制ROC曲线
                fpr,tpr,threshold = roc_curve(matching_label, matching_out) ###计算真正率和假正率  
                
                q = np.where(0.95<=tpr )
                q_value = q[0][0]
                fpr95 = fpr[q_value]
                
                save_file.write('match fpr95 : '+str(fpr95*100))
                save_file.write('\n')
                save_file.write('\n')
                save_file.close()
                roc_dir = 'plot_curve_pro/epoch_'+str(number)
                try: 
                    os.makedirs(roc_dir)
                except os.error:
                    pass
                np.savez(roc_dir+'/match_map_gd_pro_'+file_name, fpr,tpr)
            except KeyboardInterrupt:
                print('INTERRUPTED')
                coord.request_stop()
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)
                
'''
test all datasets
'''
def mc_test_all(number):
  test_data = ['field', 'forest','indoor','mountain','oldbuilding','street','urban','water']
  for i in range(8):
    mc_test(test_data[i],number)

if __name__ == '__main__':
    
#    mc_train()
#    mc_test('country',0)

    mc_test_all(1)
    
    
        
