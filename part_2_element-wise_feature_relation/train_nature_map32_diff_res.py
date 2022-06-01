#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2018.11.9

@author: wrj
"""
'''
双分支 + 特征map32 + 交叉熵损失函数 + 参数共享 + 上支路差值求绝对值
'''

import numpy as np
import math
import time
import model_d_map32_diff_res as model
import read_data
import tensorflow as tf
import os
from datetime import datetime
import logging
from sklearn.metrics import roc_curve, auc

os.environ['CUDA_VISIBLE_DEVICES']='0'

BATCH_SIZE_matching=100
epoch = 40
learning_rate = 2e-4
image_width = 64
image_height = 64

checkpoint_dir = 'ckpt_map32_diff_res'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
train_dir='summary'

def initLogging(logFilename='record_map32_diff_res.log'):
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
    
    data1 = np.load('/home/ws/文档/wrj/data_all/country.npz')
    train_matching_y = data1['arr_1'][:,np.newaxis]
    numbers_train = train_matching_y.shape[0]  #训练集总数
    epoch_steps = np.int(numbers_train / BATCH_SIZE_matching)  +1  # 一个epoch有多少个steps

    all_loss = np.array([])
    graph = tf.Graph()
    with graph.as_default():
        inputs_p1 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width,1], name='inputs_p1')
        inputs_p2 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width,1], name='inputs_p2')
        label_m = tf.placeholder(tf.float32, [BATCH_SIZE_matching, 1], name='label_m')

        # 训练 M
        match_loss, match_loss_res, match_output, match_output_res, all_features = model.match_network_res(inputs_p1, inputs_p2, label_m)
        match_out = tf.round(match_output)
        m_correct,m_numbers = model.evaluation(match_out, label_m)
        match_out_res = tf.round(match_output_res)
        m_correct_res,m_numbers_res = model.evaluation(match_out_res, label_m)
        
        var_match = [var for var in tf.trainable_variables() if var.name.startswith("matching")]
        var_matching_res = [var for var in tf.trainable_variables() if var.name.startswith("match_res")]
        var_matching_feature = [var for var in tf.trainable_variables() if var.name.startswith("matching/feature_layers")]
        var_res = var_matching_feature + var_matching_res  # modify feature layers and softmax layers
        
        m_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(match_loss, var_list=var_match)
        r_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(match_loss_res, var_list=var_res)
        

        filename = '/home/ws/文档/wrj/data_all/country/country.tfrecord'
        filename_queue = tf.train.string_input_producer([filename],num_epochs=epoch,shuffle=True)
        img_batch, label_batch = read_data.batch_inputs(filename_queue,  train = True, batch_size = BATCH_SIZE_matching)
        tf.summary.scalar('mathing_loss', match_loss)
        tf.summary.scalar('match_loss_res', match_loss_res)

        summary = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=20)
        init = tf.global_variables_initializer()
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        
        with tf.Session(config=sess_config) as sess:
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
                        feed_dict = {inputs_p1:x_batch, inputs_p2:y_batch, label_m:l_batch}
                        _,_, m_loss, m_loss_res, m_output_, m_output_res_ = sess.run([m_train_opt, r_train_opt, match_loss, match_loss_res, 
                                                                                      match_output, match_output_res], feed_dict = feed_dict)
                        
                        if step % 10 ==0:
                            loss_write = np.array([[step, m_loss, m_loss_res]])
                            if step ==10:
                                all_loss = loss_write
                            else:
                                all_loss = np.concatenate((all_loss, loss_write))
                        
                        if step % 100 == 0:
                            duration = time.time() - start_time
                            summary_str = sess.run(summary, feed_dict=feed_dict)
                            summary_writer.add_summary(summary_str, step)
                            summary_writer.flush()
                            logging.info('>> Step %d run_train: matching_loss = %.3f matching_loss_res = %.3f  (%.3f sec)'
                                          % (step, m_loss, m_loss_res, duration))
                            
                        if (step % epoch_steps == 0) and ((step / epoch_steps)%3 == 0):
                            current_epoch = int(step / epoch_steps)
                            
                            if current_epoch > 20:
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
                np.save(os.path.join(checkpoint_dir,'ckpt_map32_diff_res'), all_loss)
                print('Model saved in file :%s'%checkpoint_dir)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)
            
'''
test the network
'''
def mc_test(file_name, number):
    data = np.load('/home/ws/文档/wrj/data_all/'+file_name+'.npz')
    train_matching_y = data['arr_1'][:,np.newaxis]
    numbers_train = train_matching_y.shape[0]  #训练集总数
    
    graph = tf.Graph()
    with graph.as_default():
        inputs_p1 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width, 1], name='inputs_p1')
        inputs_p2 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width, 1], name='inputs_p2')
        label_m = tf.placeholder(tf.float32, [BATCH_SIZE_matching, 1], name='label_m')

        match_loss, match_loss_res, match_output, match_output_res, all_features = model.match_network_res(inputs_p1, inputs_p2, label_m)
        match_out = tf.round(match_output)
        m_correct,m_numbers = model.evaluation(match_out, label_m)
        match_out_res = tf.round(match_output_res)
        m_correct_res,m_numbers_res = model.evaluation(match_out_res, label_m)
        
        filename = '/home/ws/文档/wrj/data_all/country/'+file_name+'.tfrecord'
        filename_queue = tf.train.string_input_producer([filename],num_epochs=1,shuffle=False)
        img_batch, label_batch = read_data.batch_inputs(filename_queue,  train = False, batch_size = BATCH_SIZE_matching)
        
        saver = tf.train.Saver()
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.local_variables_initializer())
#            saver.restore(sess, tf.train.latest_checkpoint('ckpt_map32_diff_res'))
            saver.restore(sess, 'map_diff_res/model.ckpt-30')
            m_count = 0  # Counts the number of correct predictions.
            r_count = 0
            num = numbers_train
            matching_out = np.array([])
            matching_out_res = np.array([])
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
                        m_correct_, m_correct_res_, m_output_, m_output_res_ = sess.run([m_correct, m_correct_res, match_output, match_output_res], feed_dict = feed_dict)
                            
                        if step_test == 1:
                            matching_out = m_output_
                            matching_out_res = m_output_res_
                            matching_label = l_batch
                        elif(l_batch.size == BATCH_SIZE_matching):
                                matching_out = np.concatenate((matching_out, m_output_))
                                matching_out_res = np.concatenate((matching_out_res, m_output_res_))
                                matching_label = np.concatenate((matching_label, l_batch))
                                
                        m_count = m_count + m_correct_.astype(int)
                        r_count = r_count + m_correct_res_.astype(int)
                        if step_test % 100 == 0:
                            print('Step %d run_test: batch_precision = %.2f  batch_precision_res = %.2f '% (step_test, m_correct_/BATCH_SIZE_matching, m_correct_res_/BATCH_SIZE_matching))
                except Exception as e:
                    coord.request_stop(e)
                m_precision = float(m_count) / num
                r_precision = float(r_count) / num

                print('  Num examples: %d  Num correct: %d  match Precision : %0.04f  match_res Precision : %0.04f' %(num, m_count, m_precision, r_precision))
                save_file = open('test_map32_diff_res.txt','a')
                save_file.write(file_name + '    epoch: ' + str(number) +'\n'+'num correct: '+str(m_count)+'/'+str(num)+'     match precision : '+str(m_precision))
                save_file.write('     res_num correct: '+str(m_count)+'/'+str(num)+'     match_res precision : '+str(m_precision))
                save_file.write('    ')

#               计算 tp,tn,fp,fn 
                pre_out = np.round(matching_out_res)
                pre_out = pre_out.astype(int)
                lab_out = matching_label.astype(int)
                tp = np.sum(pre_out & lab_out)
                tn = r_count - tp
                fp = np.sum(pre_out) - tp
                fn = np.sum(lab_out) - tp
                save_file.write(file_name + '    tp: ' + str(tp)+ '    tn: ' + str(tn) \
                                            + '    fp: ' + str(fp)+ '    fn: ' + str(fn))
                
        #       绘制ROC曲线
                fpr,tpr,threshold = roc_curve(matching_label, matching_out) ###计算真正率和假正率  
                fpr_res,tpr_res,threshold = roc_curve(matching_label, matching_out_res)
                
                q = np.where(0.95<=tpr )
                q_value = q[0][0]
                fpr95 = fpr[q_value]
                q_res = np.where(0.95<=tpr_res )
                q_value_res = q_res[0][0]
                fpr95_res = fpr_res[q_value_res]
                
                save_file.write('match_diff fpr95 : '+str(fpr95*100)+'     match_diff_res fpr95 : '+str(fpr95_res*100))
                save_file.write('\n')
                save_file.write('\n')
                save_file.close()
                roc_dir = 'plot_curve/roc/nature_map32_diff_res/epoch_'+str(number)
                try:
                    os.makedirs(roc_dir)
                except os.error:
                    pass
                np.savez(roc_dir+'/match_map32_diff_res_match_'+file_name, fpr,tpr)
                np.savez(roc_dir+'/match_map32_diff_res_res_'+file_name, fpr_res,tpr_res)
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
#    mc_test('field',1)
    mc_test_all(1)
    
    
        
