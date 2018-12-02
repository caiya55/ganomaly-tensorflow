# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:22:18 2018

@author: zhang
"""

from __future__ import print_function
from tensorflow.python.keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import math
from keras.utils. generic_utils import Progbar
from options import get_config
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

''' calculate the auc value for lables and scores'''
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

'''resize the images in batches'''
def batch_resize(imgs, size):
    img_out = np.empty([imgs.shape[0], size[0], size[1]])
    for i in range(imgs.shape[0]):
        img_out[i] = cv2.resize(imgs[i], size, interpolation=cv2.INTER_CUBIC)
    return img_out
''' type-1 loss and type2 loss'''
def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def bce_loss(y_pred,y_true):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                                  logits=y_pred))
#    return -tf.reduce_mean(y_true * tf.log(y_pred)+(1-y_true)* tf.log(1-y_pred))
'''Tensorflow based bat normalizatioin'''
def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
  return tf.contrib.layers.batch_norm(x, 
    decay=momentum,
    updates_collections=None,
    epsilon=epsilon,
    scale=True,
    is_training=is_train, 
    scope=name)

'''generate the model is based on 
    https://arxiv.org/abs/1805.06725
    gerneator includes Encoder->Decoder->Encoder
 '''
def Encoder(inputs, opts, istrain=True, name='e1'):
    assert opts.isize % 16 == 0, "isize has to be a multiple of 16"
    ''' initial layer'''
    x = Conv2D(opts.gen_filter, (4, 4), strides=2, padding='same', use_bias=False)(inputs)
    x = LeakyReLU(0.2)(x)
    size_now = opts.isize // 2
    ''' Extra layers'''
    for t in range(opts.n_extra_layers):
        x = Conv2D(opts.gen_filter, (3, 3), padding='same', use_bias=False)(x)
        x = batch_norm(x, name+"_bn1_"+str(t), is_train=istrain)
        x = LeakyReLU(0.2)(x)

    channel = opts.gen_filter # channel: default number is 64
    ''' reduction layers'''
    while size_now > 4:
        x = Conv2D(channel*2, (4, 4), strides=2, padding='same', use_bias=False)(x)
        x = batch_norm(x, name+"_bn2_"+str(channel), is_train=istrain)
        x = LeakyReLU(0.2)(x)    
        channel = channel*2
        size_now = size_now // 2
        
    # state size. channel x 4 x 4        
    ''' final layer, resize the layer to channel X 1 X 1'''
    output = Conv2D(opts.z_size, (4, 4), padding='valid', use_bias=False)(x)
            
    return output

def Decoder(inputs, opts,istrain=True):
    assert opts.isize % 16 == 0, "isize has to be a multiple of 16"
    cngf, tisize = opts.dis_filter // 2, 4
    while tisize != opts.isize:
        cngf = cngf * 2  # after loop, cngf reaches to 256
        tisize = tisize * 2
    '''z is input, and first deconvolution layer to size channel * 4 * 4'''
    x = Conv2DTranspose(cngf, (4, 4), padding='valid', use_bias=False )(inputs)
    x = batch_norm(x, "bn1", is_train=istrain)
    x = Activation('relu')(x)   
    '''size increaing layers '''
    size_now = 4
    while size_now < opts.isize // 2:
        x = Conv2DTranspose(cngf//2, (4, 4),strides=2, padding='same', use_bias=False )(x)
        x = batch_norm(x, "bn2_"+str(size_now), is_train=istrain)
        x = Activation('relu')(x)      
        cngf = cngf // 2
        size_now = size_now*2
    '''extral layers, keep the channel and size of the layers same'''
    for t in range(opts.n_extra_layers):
        x = Conv2DTranspose(cngf, (3, 3), padding='same', use_bias=False )(x)
        x = batch_norm(x, "bn3_"+str(t), is_train=istrain)
        x = Activation('relu')(x)   
    ''' final layer, expand the size with 2 and channel of n_output_channel'''
    x = Conv2DTranspose(opts.image_channel, (4, 4),strides=2, padding='same', use_bias=False )(x)
    x = Activation('tanh')(x)       
    return x
'''generator is the encoder->decoder->encoder structure'''
def generator(inputs, opts, istrain=True):
    with tf.variable_scope('gen_'):
        z      = Encoder(inputs, opts, istrain=istrain, name='e1')
        x_star = Decoder(z, opts, istrain=istrain)
        z_star = Encoder(x_star, opts, istrain=istrain, name='e2')
    return x_star, z, z_star


''' discriminator is the same as discriminator from DCGan
    the last layer is utilized for feature mapping loss
    in the original code, the dis net is the same as the encoder without final layer
'''
def discriminator(inputs, opts, reuse=False, istrain=True, name='d1'):
    with tf.variable_scope('dis_', reuse=reuse):
#        x = Conv2D(64, (5, 5), padding='same')(inputs)
#        x = Activation('tanh')(x)  
#        x = MaxPooling2D(pool_size=(2, 2))(x)
#        x = Conv2D(128, (5, 5))(x)
#        x = Activation('tanh')(x)
#        x = MaxPooling2D(pool_size=(2, 2))(x)
#        x = Flatten()(x)
#        x = Dense(1024)(x)
#        feature = Activation('tanh')(x)
#        x = Dense(1)(feature)
#        x = Activation('sigmoid')(x)
#        return feature, x
        ''' initial layer'''
        x = Conv2D(opts.gen_filter, (4, 4), strides=2, padding='same', use_bias=False)(inputs)
        x = LeakyReLU(0.2)(x)
        size_now = opts.isize // 2
        ''' Extra layers'''
        for t in range(opts.n_extra_layers):
            x = Conv2D(opts.gen_filter, (3, 3), padding='same', use_bias=False)(x)
            x = batch_norm(x, name+"_bn1_"+str(t), is_train=istrain)
            x = LeakyReLU(0.2)(x)
    
        channel = opts.gen_filter # channel: default number is 64
        ''' reduction layers'''
        while size_now > 4:
            x = Conv2D(channel*2, (4, 4), strides=2, padding='same', use_bias=False)(x)
            x = batch_norm(x, name+"_bn2_"+str(channel), is_train=istrain)
            x = LeakyReLU(0.2)(x)    
            channel = channel*2
            size_now = size_now // 2
        feature = x
        # state size. channel x 4 x 4        
#        ''' final layer, resize the layer to channel X 1 X 1'''
        x = Conv2D(1, (4, 4), padding='valid', use_bias=False)(x)
        x = Flatten()(x)
        classifier = Dense(1)(x)
#        x = tf.reshape(x, [-1])
#        classifier = Activation('sigmoid')(x)
        return feature, classifier
        
#        classifier = Activation('sigmoid')(x)
#        classifier = x# dont use sigmoid in last layer               

'''discriminator and generator together class'''

class Ganormal(object):
    def __init__(self, sess, opts):                  
       self.sess = sess
       self.is_train = tf.placeholder(tf.bool)
       self.imsize = opts.isize
       self.im_shape = [opts.batch_size, opts.isize, opts.isize, 1]
       self.img_input = tf.placeholder(tf.float32, self.im_shape)
       self.opts = opts
       ''' 0 create model'''   
       with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
           self.img_gen, self.latent_z, self.latent_z_gen = generator(self.img_input, self.opts, self.is_train)
           self.feature_fake, self.label_fake = discriminator(self.img_gen, self.opts,False, self.is_train)
           self.feature_real, self.label_real = discriminator(self.img_input,self.opts,True, self.is_train)
       self.t_vars = tf.trainable_variables()
       self.d_vars = [var for var in self.t_vars if 'dis_' in var.name]
       self.g_vars = [var for var in self.t_vars if 'gen_' in var.name]
       '''1 create losses'''
       self.adv_loss = l2_loss(self.feature_fake,self.feature_real )
#       self.adv_loss = bce_loss(self.label_fake, tf.ones_like(self.label_fake))
       self.context_loss = l1_loss(self.img_input, self.img_gen)
       self.encoder_loss = l2_loss(self.latent_z, self.latent_z_gen)
       self.generator_loss = 0.5*self.adv_loss +50*self.context_loss + 1*self.encoder_loss
       '''dis loss: real label reach to 1 and fake label reach to 0'''
       self.real_loss = bce_loss(self.label_real, tf.ones_like(self.label_real)) # real reach to 1
       self.fake_loss = bce_loss(self.label_fake, tf.zeros_like(self.label_fake))
       self.feature_loss = self.real_loss + self.fake_loss #-l2_loss(self.feature_fake, self.feature_real)#
       self.discriminator_loss = self.feature_loss
       '''2 optimize the loss, learning rate and beta1 is from Original code of Pytorch '''
       update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
       with tf.control_dependencies(update_ops):
           with tf.variable_scope(tf.get_variable_scope(), reuse=None):
               self.gen_train_op = tf.train.AdamOptimizer(
                   learning_rate=2e-3,beta1=0.5,beta2=0.999).minimize(self.generator_loss,var_list=self.g_vars)
               self.dis_train_op = tf.train.AdamOptimizer(
                   learning_rate=2e-3,beta1=0.5,beta2=0.999).minimize(self.discriminator_loss,var_list=self.d_vars)
       '''3 save the model '''    
       self.saver = tf.train.Saver()
       '''4 initialization'''
       self.sess.run(tf.global_variables_initializer())

    ''' generator training in keras style'''
    def gen_fit(self, batch_x):    
        _, loss,al,cl,el  = self.sess.run([self.gen_train_op, 
                                  self.generator_loss,
                                  self.adv_loss,
                                  self.context_loss,
                                  self.encoder_loss], 
                               {self.img_input:batch_x,self.is_train: True,})
        return loss,al,cl,el
    
    ''' discriminator training in keras style'''
    def dis_fit(self, batch_x):     
        _, loss,dis_real_loss, dis_fake_loss  = self.sess.run([self.dis_train_op, self.discriminator_loss,
                                  self.real_loss,
                                  self.fake_loss], 
          {self.img_input:batch_x,self.is_train: True,})
        return loss, dis_real_loss, dis_fake_loss
    ''' train the model in dis and gen'''
    def train(self, batch_x):
        gen_loss,al,cl,el = self.gen_fit(batch_x)
        _, dis_real_loss, dis_fake_loss  = self.dis_fit(batch_x)
        # If D loss is zero, then re-initialize netD
        if dis_real_loss < 1e-5 or dis_fake_loss < 1e-5:    
            init_op = tf.initialize_variables(self.d_vars)
            self.sess.run(init_op)
#            print('reinitialize')
        return gen_loss, al,dis_real_loss,dis_fake_loss
    def evaluate(self, whole_x, whole_y):
        bs = self.opts.test_batch_size
        labels_out, scores_out = [], []     
        index = 1
        for index in range(int(whole_x.shape[0] / bs)):
            batch_x = whole_x[index*bs:(index+1)*bs]
            batch_y = whole_y[index*bs:(index+1)*bs]
            latent_loss, latent_gen_loss = self.sess.run([self.latent_z, 
                                                          self.latent_z_gen], 
                                 {self.img_input:batch_x,self.is_train: False,})
            latent_error = np.mean(abs(latent_loss - latent_gen_loss), axis=-1) 
            latent_error = np.reshape(latent_error, [-1])
            scores_out = np.append(scores_out, latent_error)
            labels_out = np.append(labels_out, batch_y)
            ''' Scale scores vector between [0, 1]'''
            scores_out = (scores_out - scores_out.min())/(scores_out.max()-scores_out.min())
        '''calculate the roc value'''
        auc_out = roc(labels_out, scores_out)
        return scores_out, labels_out, auc_out

    def save(self, dir_path):
        self.saver.save(self.sess, dir_path+"/model.ckpt")
    '''show the generated images'''
    def show(self,single_x):
        generated_img = self.sess.run(self.img_gen, {self.img_input:single_x, self.is_train:False})
        plt.imshow(generated_img[0,:,:,0])
        plt.show()
        plt.imshow(single_x[0,:,:,0])
        plt.show()
        return generated_img[0,:,:,0], single_x[0,:,:,0]
if __name__ == "__main__":  
    opts = get_config(is_train=True)    
    inputs = tf.placeholder(tf.float32, [None, 32, 32, 1])    
#    gen_test,z_test,z_star_test = generator(inputs, opts, istrain=True)
    feature_test, dis_test = discriminator(inputs, opts,tf.AUTO_REUSE,True)
#    print(gen_test)
    print(feature_test, dis_test)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    