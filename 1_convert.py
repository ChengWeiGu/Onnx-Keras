# test a fig by the command : python .\predict.py --weights="ENetB0_4cls.h5"  --image="OK-N2-g.bmp"

import time
import numpy as np
import tensorflow as tf

from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten , Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, losses
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint


import cv2
import os
os.environ['TF_KERAS'] = '1'
from os import walk, listdir
from os.path import basename, dirname, isdir, isfile, join
import json
import argparse

# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import keras2onnx

class effnet:

    def __init__(self):
        
        num_cls = 5
        imag_size = 224
        #------------------------------------start building model-----------------------------# 
        model = EfficientNetB0(weights = 'imagenet', input_shape = (imag_size,imag_size,3), include_top = False)

        ENet_out = model.output
        ENet_out = Flatten()(ENet_out)

        Hidden1_in = Dense(1024, activation="relu")(ENet_out)
        Hidden1_in = Dropout(0.5)(Hidden1_in)

        predictions = Dense(units = num_cls, activation="softmax")(Hidden1_in)
        self.model_f = Model(inputs = model.input, outputs = predictions)
        self.model_f.compile(optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy])
        self.model_f.load_weights("efficientnetB0_weights.h5")
        # self.model_f.summary()
        # print(self.model_f.name)
        

if __name__ == '__main__':

    net = effnet()
    
    save_model_file = "panel.onnx"
    model = net.model_f
    # print(model.inputs)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    keras2onnx.save_model(onnx_model, save_model_file)
