## The efficientnet model is tested for onnx transformation:

## Model Installation:
Please refer to the following webstite:  
https://github.com/qubvel/efficientnet  

## Installation and Environment:   
1.GPU: MX110 2G  
2.tensorflow-gpu2.1.0  
3.tensorflow1.14.0  
4.keras2.2.4  
5.keras2onnx1.7.0  
6.onnxruntime1.2.0
7.cudnn7.6.5  
8.efficientnet1.1.0   


## Run the scripts:
Step1. before runing the scripts, you need to prepare the h5 files where model and its weights are saved.  
And then by the following command:   

python 1_convert.py  

there will be a corresponding onnx file generated.   

Step2. After the first step, one can implement the onnx file by the next script.  
that is,  

python 2_inference.py  

In this case, the efficientnetB0 is utilized.  



