## The efficientnet model is tested for onnx transformation:

## Model Installation:
Please refer to the following webstite:  
https://github.com/qubvel/efficientnet  

## Installation and Environment:   
1.GPU: MX110 2G  
2.tensorflow-gpu==2.1.0  
3.tensorflow==1.14.0  
4.keras==2.2.4  
5.keras2onnx==1.7.0  
6.onnxruntime==1.2.0  
7.cudnn==7.6.5    
8.efficientnet==1.1.0    


## Run the scripts:
#Step1:  
Before runing the scripts, you need to prepare the h5 files where model and its weights are saved.  
And then by the following command:   

python 1_convert.py  

there will be a corresponding onnx file generated.   

#Step2:  
After the first step, one can implement the onnx file by the next script. That is,   

python 2_inference.py  

In this case, the efficientnetB0 is utilized.   

## Run the merged script:  
one can run the merged version of onnx from convertion to inference by    
python keras_onnx.py



