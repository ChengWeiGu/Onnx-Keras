from os.path import basename, join
from os import listdir
import numpy as np
from efficient.Im_prepro import preprocess_imgs # the image preprocess package not provided here
import time 
import onnxruntime

def image_unit_test(image, onnx_sess):
    
    t0 = time.time()
    test_predictions = onnx_sess.run([output_name], {input_name: image})[0]
    t = time.time() - t0
    
    y_test_pre = np.argmax(test_predictions,axis = 1)
    res_class = class_names[str(y_test_pre[0])] #3/24 added
    print("the img name: {}; the predicted class: {}".format(basename(image_path),res_class)) #3/24 added
    
    return res_class, t

if __name__ == "__main__":
    
    onnx_model_path = "./panel.onnx"
    class_names = {'0':'OK', '1':'NG-WL','2':'NG-BL','3':'NG-SP','4':'NG-LL'}
    
    sess = onnxruntime.InferenceSession(onnx_model_path)
    sess.set_providers(['CPUExecutionProvider'])
    
    # input
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    # output
    output_name = sess.get_outputs()[0].name   

    image_dir = "./image_data"
    
    
    total_time = 0
    for filename in listdir(image_dir):        
        image_path = join(image_dir, filename)
        imag_bgr_small, imag_c_small = preprocess_imgs(image_path) 
        tests = imag_c_small.reshape(-1,input_shape[1],input_shape[2],3)
        tests = tests.astype('float32') / 255
        res_class, t = image_unit_test(tests, sess)
        total_time += t
    print("Time Elapsed: {:.4f} sec./pic".format(total_time/len(listdir(image_dir))))
       
