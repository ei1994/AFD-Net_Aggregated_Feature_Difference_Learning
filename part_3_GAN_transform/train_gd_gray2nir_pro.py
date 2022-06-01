
# -*- coding: utf-8 -*-
"""
Created on 2018.11.18

@author: DELL
"""

'''
用于生成器的训练，将九个数据集一起训练一个生成器
gray --> nir
'''

import numpy as np
import math
import time 
import model_gd_diff_pro as model
import tensorflow as tf
import read_data_map as read
import os
from datetime import datetime
import logging
import cv2

BATCH_SIZE_matching = 100
EPOCH = 81
learning_rate = 2e-4
image_width = 64
image_height = 64
#checkpoint_dir = 'ckpt_gd_o2r'
#checkpoint_dir_g = 'ckpt_g_o2r'
#output_dir = 'out_o2r'
train_dir='summary_gd'

os.environ['CUDA_VISIBLE_DEVICES']='0'

def initLogging(logFilename='record_gd_g2r.log'):
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

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:,:,0]
    return image

def preprocess(image):
#    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2.0 - 1.0

def deprocess(image):
#    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1.0) / 2.0

def gd_train(f_name):
    checkpoint_dir = 'ckpt_gd_g2r_pro_'+f_name
    checkpoint_dir_g = 'ckpt_g_g2r_pro_'+f_name
    output_dir = 'out_g2r_pro_'+f_name
    checkpoint_dir_en = 'ckpt_en_r2r_'+f_name
    
    checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
    checkpoint_file_g = os.path.join(checkpoint_dir_g, 'model.ckpt')
    initLogging('record_gd_g2r_pro_'+f_name+'.log')
#    if FLAGS.load_model is not None:
#        checkpoints_dir = 'checkpoints/' + FLAGS.load_model
#    else:
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    try:
        os.makedirs(checkpoint_dir_g)
        os.makedirs(checkpoint_dir)
    except os.error:
        pass
    try:
        os.makedirs(output_dir)
    except os.error:
        pass
#    
    data1 = np.load('/home/ws/文档/wrj/mapping_data_gray/map_' + f_name + '.npz')
    train_matching_y = data1['arr_0'][:,np.newaxis]
    numbers_train = train_matching_y.shape[0]  #训练集总数
    epoch_steps = np.int(numbers_train / BATCH_SIZE_matching)  +1  # 一个epoch有多少个steps

    all_loss = np.array([])
    graph = tf.Graph()
    with graph.as_default():
        inputs_p1 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width,1], name='inputs_gray')
        inputs_p2 = tf.placeholder(tf.float32, [BATCH_SIZE_matching, image_height, image_width,1], name='inputs_nir')
#        test_opt = tf.placeholder(tf.float32, [None, image_height, image_width, 1], name='test_opt')

       # 训练 G
       #    将输入图像从 （0，1）变换到 （-1，1）
        inputs_p1_ = model.preprocess(inputs_p1)
        inputs_p2_ = model.preprocess(inputs_p2)
        gen_loss, dis_loss, _, _ = model.gd_model_g2r(inputs_p1_, inputs_p2_)
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(dis_loss,var_list=discrim_tvars)
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss,var_list=gen_tvars)
        nirencode_tvars = [var for var in tf.trainable_variables() if var.name.startswith("nir_encode")]
        
#        with tf.control_dependencies([g_train_opt, d_train_opt]):
#             gd_train_opt = tf.no_op(name='optimizers') 
        
#        
        filename = '/home/ws/文档/wrj/mapping_data_gray/map_data/map_' + f_name + '.tfrecord'
        filename_queue = tf.train.string_input_producer([filename],num_epochs=EPOCH,shuffle=True)
        img_batch, label_batch = read.batch_inputs(filename_queue, train = True, batch_size = BATCH_SIZE_matching)

        saver = tf.train.Saver(max_to_keep=20)
        saver_g = tf.train.Saver(var_list=gen_tvars, max_to_keep=20)
        saver_en = tf.train.Saver(var_list=nirencode_tvars)
        init = tf.global_variables_initializer()
        #
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(init)
            saver_en.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_en))
            try:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                step = 0
                while not coord.should_stop():
                        start_time = time.time()
                        step = step + 1
                        batch, l_batch = sess.run([img_batch, label_batch])
                        x_batch = batch[:,:,:64,np.newaxis]
                        y_batch = batch[:,:,64:,np.newaxis]
                        feed_dict = {inputs_p1:x_batch, inputs_p2:y_batch}
                        _, _, g_loss, d_loss = sess.run([d_train_opt, g_train_opt, gen_loss, dis_loss], feed_dict = feed_dict)
                        
                        if step % 10 ==0:
                            loss_write = np.array([[step, g_loss, d_loss]])
                            if step ==10:
                                all_loss = loss_write
                            else:
                                all_loss = np.concatenate((all_loss, loss_write))
                        
                        if step % 100 == 0:
                            duration = time.time() - start_time
                            logging.info('>> Step %d run_train: g_loss = %.2f  d_loss = %.2f (%.3f sec)'
                                      % (step, g_loss, d_loss, duration))
                            
                        if (step % epoch_steps == 0) and ((step / epoch_steps)%10 == 0):
                            current_epoch = int(step / epoch_steps)
                            
                            if current_epoch >= 40:
                              logging.info('>> %s Saving in %s' % (datetime.now(), checkpoint_dir))
                              saver.save(sess, checkpoint_file, global_step=current_epoch)
                              saver_g.save(sess, checkpoint_file_g, global_step=current_epoch)
#                              mc_test_all(current_epoch)
                            if current_epoch >= 40:
                                mc_test(f_name, current_epoch)
                              
            except KeyboardInterrupt:
                print('INTERRUPTED')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)

            finally:
                saver.save(sess, checkpoint_file, global_step=step)
                saver_g.save(sess, checkpoint_file_g, global_step=step)
                np.save(os.path.join(checkpoint_dir,'ckpt_map_gd_pro_'+f_name), all_loss)
                np.save(os.path.join(checkpoint_dir_g,'ckpt_map_gd_pro_'+f_name), all_loss)
                print('Model saved in file :%s'%checkpoint_dir)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)

'''
test the network
'''
def mc_test(file_name, number):

#    data = np.load('/home/ws/文档/wrj/mapping_data_all/map_rgb_'+file_name+'.npz')
#    train_matching_y = data['arr_0'][:]
#    numbers_train = train_matching_y.shape[0]  #训练集总数
    BATCH_SIZE_map = 40
    
    graph = tf.Graph()
    with graph.as_default():
        inputs_p1 = tf.placeholder(tf.float32, [BATCH_SIZE_map, image_height, image_width, 1], name='inputs_gray')
        inputs_p2 = tf.placeholder(tf.float32, [BATCH_SIZE_map, image_height, image_width, 1], name='inputs_nir')
    
        gen_loss, dis_loss, _, _ = model.gd_model_g2r(inputs_p1, inputs_p2)
        inputs_p1_ = model.preprocess(inputs_p1)
        gen = model.create_generator(inputs_p1_, 1, reuse=True)
        gen = model.deprocess(gen)
        
#        filename = '/home/ws/文档/wrj/mapping_data_gray/map_data/map_'+file_name+'.tfrecord'
        filename = '/home/ws/文档/wrj/data_all/country/'+file_name+'.tfrecord'
        filename_queue = tf.train.string_input_producer([filename],num_epochs=1,shuffle=False)
        img_batch, label_batch = read.batch_inputs(filename_queue, train = False, batch_size = BATCH_SIZE_map)
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
#            saver.restore(sess, tf.train.latest_checkpoint('ckpt_gd_g2r_pro_all'))
            saver.restore(sess, 'ckpt_gd_g2r_pro_all/model.ckpt-' + str(number))
#            num = 0
            
            try:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                step_test = 0
                while not coord.should_stop():
                    if step_test < 1:
                        
                        step_test = step_test + 1
                        batch, l_batch = sess.run([img_batch, label_batch])
                        x_batch = batch[:,:,:64,np.newaxis]
                        y_batch = batch[:,:,64:,np.newaxis]
                        feed_dict = {inputs_p1:x_batch}
                        gen_out = sess.run(gen, feed_dict = feed_dict)
                        
                        gen_out_dir = 'out_g2r_pro_all/epoch_' + str(number)
                        try:
                            os.makedirs(gen_out_dir)
                        except os.error:
                            pass
                        show_images = np.concatenate((y_batch, gen_out), axis=1)
                        show_images = show_images*255
                        for i in range(BATCH_SIZE_map):
                            cv2.imwrite(gen_out_dir+'/{}.png'.format(file_name+'_'+str(i+1)+"_gray"), np.squeeze(x_batch[i,:,:,:])*255)
                            cv2.imwrite(gen_out_dir+'/{}.png'.format(file_name+'_'+str(i+1)+"_nir"), np.squeeze(show_images[i,:64,:,:]))
                            cv2.imwrite(gen_out_dir+'/{}.png'.format(file_name+'_'+str(i+1)+"_fnir"), np.squeeze(show_images[i,64:,:,:]))
                    else:
                        break
            except KeyboardInterrupt:
                print('INTERRUPTED')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

'''
test all datasets
'''
def mc_test_all(number):
  
  test_data = ['country', 'field', 'forest','indoor','mountain','oldbuilding','street','urban','water']
  for i in range(9):
    mc_test(test_data[i],number)
    
def mc_train_all():
#  'country', 
  train_data = ['country', 'field', 'forest','indoor','mountain']
  for i in range(5):
    gd_train(train_data[i])
            
if __name__ == '__main__':
#    mc_train_all()
  
#  gd_train('all')
  for i in [70,80]:
    mc_test_all(i)
#  mc_test('country', 25)

