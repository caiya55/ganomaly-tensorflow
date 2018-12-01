# -*- coding: utf-8 -*-
"""
Created on 11/29/2018

@author: zhang

This a template code for the implement of Ganormal based on tensorflow 1.5.0. 

test dataset is mnist. one class is considered as abormal, and others are considered as normal.
Actually this abnormal-detection idea is very good for the claissification missions while the 
training dataset is very imbalanced.
All right reseveved by Laboro.ai
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import tensorflow as tf
from ganormal import Ganormal, batch_resize
from options import get_config
from tqdm import tqdm

if __name__ == "__main__":  
    ''' 0. prepare mnist data '''
    anomaly = 2
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    '''normalized the dataset'''
    x_train = (x_train.astype(np.float32) / 255.0 - 0.1307) / 0.3081
    x_test = (x_test.astype(np.float32) / 255.0 - 0.1307) / 0.3081
    
    ''' training takes only numbers without n, here n default is 2, because number 2
        shows the best performance in the manuscript
        and test takes every numbers. and set label-2 class as 1, other sest is to 0.
        resize the images into 32*32 size in order to fit the model input
    '''
    x_train = x_train[y_train != anomaly] 
    x_train = batch_resize(x_train,(32,32)) 
    x_test = batch_resize(x_test,(32,32))
    y_test = 1*(y_test==anomaly)
    '''add one new axis to fit the model input'''
    x_train = x_train[:,:,:,None]
    x_test = x_test[:,:,:,None]

    print ('train shape:', x_train.shape)
    
    ''' 1. train model and evaluate on test data by AUC '''
    sess = tf.Session()   
    opts = get_config(is_train=True)    
    model = Ganormal(sess, opts)
    ''' 
    strat training
    '''    
    auc_all = []
    for i in range(opts.iteration):
        loss_train_all = []
        loss_test_all = []
        real_losses = []
        fake_losses = []
        enco_losses = []
        ''' shuffle data in each epoch'''
        permutated_indexes = np.random.permutation(x_train.shape[0])
        ''' decay the learning rate. we dont do that in tensorflow way because it
        is more easier to fine-tuning'''
        for index in tqdm(range(int(x_train.shape[0] / opts.batch_size))):
            batch_indexes = permutated_indexes[index*opts.batch_size:(index+1)*opts.batch_size]
            batch_x = x_train[batch_indexes]
            loss, al,cl,el = model.train(batch_x)
            loss_train_all.append(loss)
            real_losses.append(al)
            fake_losses.append(cl)
            enco_losses.append(el)
        print("iter {:>6d} :{:.4f} a:{:.4f} c {:.4f} e{:.4f}".format(i+1, np.mean(loss_train_all),
                                                  np.mean(al),
                                                  np.mean(cl),
                                                  np.mean(el)))
        g1, r1 = model.show(batch_x)            
        ''' evaluate the model with test dataset and calculate the AUC value'''
        if (i+1) % 1 == 0:
            scores_out, labels_out, auc_out = model.evaluate(x_test, y_test)
            print("iter {:>6d} :AUC {}".format(i+1, auc_out))
            auc_all.append(auc_out)
        '''save the model'''
        if (i+1) % 4 ==0:
            model.save(opts.ckpt_dir)
        ''' visualization'''
    plt.plot(auc_all)
    plt.xlabel('iteration')
    plt.ylabel('AUC value on test dataset')
    plt.grid(True)
    plt.show()
            
            
            
            
            
            
            
            
            
            
            
            