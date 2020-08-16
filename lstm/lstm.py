#! /usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
from keras import layers, models, optimizers
from keras.utils import to_categorical
from keras.layers import *
from keras.models import *
from keras.callbacks import Callback
from keras import backend as K
K.clear_session()
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from keras import callbacks
from keras import backend as K 
K.set_image_data_format('channels_last')

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

def cc2(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)

#parameters for LSTM
window_length = 19
rows, cols = window_length, 52
nb_lstm_outputs = 700  
nb_time_steps = window_length
nb_input_vector = cols  

def test(model, x_test, args):
    model.load_weights(args.weights)
    y_pred = model.predict(x_test, batch_size=128)
    #for i in y_pred:
        #if(i >= 1):
            #i = 0.981
        #elif(i == 0):
            #i = 0.025        
        #else:
            #i = np.around(i, decimals=3)
    print(y_pred)  
    np.savetxt("predict.txt" , y_pred) 

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    
    parser = argparse.ArgumentParser(description="LSTM stucture.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='feature_onehot_pssm')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")    
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
        
    # load data
    x_train = np.load("train_data/x_train_winlen_" + str(window_length) + ".npy")
    y_train = np.load("train_data/rasa_train_winlen_" + str(window_length) + ".npy")
    x_valid = np.load("valid_data/x_valid_winlen_" + str(window_length) + ".npy")
    y_valid = np.load("valid_data/rasa_valid_winlen_" + str(window_length) + ".npy")
    x_test = np.load("test_data/x_test_winlen_" + str(window_length) + ".npy")
    y_test = np.load("test_data/rasa_test_winlen_" + str(window_length) + ".npy")
    
    x_train = x_train.reshape(x_train.shape[0], rows, cols)
    x_valid = x_valid.reshape(x_valid.shape[0], rows, cols)
    x_test = x_test.reshape(x_test.shape[0], rows, cols)
    
    y_train = y_train.reshape(y_train.shape[0])
    y_valid = y_valid.reshape(y_valid.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    
    print(y_test)
    
    # lstm model
    model = Sequential()
    model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector), return_sequences=True))
    model.add(LSTM(units=nb_lstm_outputs))
    model.add(Dense(1, activation='softmax'))
    model.summary()
    
    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)    
    if args.testing:
        test(model=model, x_test=x_test, args=args)
    else:
        # callbacks
        log = callbacks.CSVLogger(args.save_dir + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                                   batch_size=args.batch_size, histogram_freq=int(args.debug))
        #EarlyStopping = callbacks.EarlyStopping(monitor='val_cc2', min_delta=0.01, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=True)
        #checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_mean_absolute_error', mode='min',
        checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_acc', mode='max',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
        
        #compile:loss, optimizer, metrics
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        #model.compile(loss=correlation_coefficient_loss, optimizer=optimizers.Adam(lr=args.lr), metrics=[cc2])
        #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
        
        #train: epcoch, batch_size
        model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1, callbacks=[log, tb, checkpoint], validation_data=(x_test, y_test))
        
        model.save_weights(args.save_dir + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
        
        #from utils import plot_log
        #plot_log(args.save_dir + '/log.csv', show=True)
            
     
