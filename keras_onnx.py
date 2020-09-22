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
os.environ['TF_KERAS'] = '1' # must add this lline when convert the model to onnx
import keras2onnx
import onnxruntime
import onnx


class MODEL:

    def __init__(self):
        
        self.num_cls = 5
        self.imag_size = 224
        
        model = EfficientNetB0(weights = 'imagenet', input_shape = (self.imag_size,self.imag_size,3), include_top = False)

        outputs = model.output
        outputs = Flatten()(outputs)

        Hidden1_inputs = Dense(1024, activation="relu")(outputs)
        Hidden1_inputs = Dropout(0.5)(Hidden1_inputs)

        predictions = Dense(units = self.num_cls, activation="softmax")(Hidden1_inputs)
        self.model_f = Model(inputs = model.input, outputs = predictions)
        self.model_f.compile(optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy])
        
        # self.model_f.summary()
        # print(self.model_f.name)
        

def convert():
    net = MODEL()
    net.model_f.load_weights("efficientnetB0_last.h5")
    onnx_model = keras2onnx.convert_keras(net.model_f, net.model_f.name)
    keras2onnx.save_model(onnx_model, "keras_efficientnet.onnx")
    
    
def inference(filename):
    sess = onnxruntime.InferenceSession("keras_efficientnet.onnx")
    sess.set_providers(['CPUExecutionProvider'])
    
    # input name and shape
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    # output name
    output_name = sess.get_outputs()[0].name
    
    #----------------------load image----------------------#
    img = cv2.imread(filename,0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype('float32')/255
    img = img.reshape(-1,input_shape[1],input_shape[2],3)
    #----------------------load image----------------------#
    
    outputs = sess.run([output_name], {input_name: img})[0]
    
    print("predicted result = ",outputs)
    print('finifhsed\n\n')
    
    

if __name__ == '__main__':
    convert() # step1: convert model to onnx
    inference('WL_s28.bmp') # step2: import onnx model and do prediction
    
    
    
    
    