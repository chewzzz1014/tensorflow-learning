# Train Model

## Summary
## Setup
The following setups are conducted in Ubuntu 22.10.

1. **Install Python 3**

   ```
    sudo apt update\
    sudo apt install python3
   ```
    
   Verify the installation by runnung:
   
   ```
    python3 --version
   ```
   
    If you want to use the command `python` instead of `python3` (like the rest of this documentation), install python-is-python3 package which will refer python to python3.
    
    ```
     sudo apt-get install python-is-python3
    ```
2. **Install git**

   ```
    sudo apt install git
   ```
    
   Verify the installation by runnung:
   
   ```
    git --version
   ```
   
3. **Install TensorFlow**
   
   ```
    pip install --ignore-installed --upgrade tensorflow==2.5.0
   ```
   
   Verify the installation by running:
   
   ```
    python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
   ```
   
   Expected output:
   
   ```
    2020-06-22 19:20:32.614181: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
    2020-06-22 19:20:32.620571: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    2020-06-22 19:20:35.027232: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
    2020-06-22 19:20:35.060549: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties:
    pciBusID: 0000:02:00.0 name: GeForce GTX 1070 Ti computeCapability: 6.1
    coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 8.00GiB deviceMemoryBandwidth: 238.66GiB/s
    2020-06-22 19:20:35.074967: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
    2020-06-22 19:20:35.084458: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cublas64_10.dll'; dlerror: cublas64_10.dll not found
    2020-06-22 19:20:35.094112: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
    ...
   ```
4. **Install Tensorflow Object Detection API**
     1. Create a new folder under path of your choice and name it `TensorFlow`. 
     2. cd to the `Tensorflow` folder
     3. Download the Tensorflow Models by running:
     
        ```
          # from within Tensorflow/
          git clone https://github.com/tensorflow/models
        ```
     The Tensorflow folder should look like this:
     
     ```
      TensorFlow/
      └─ models/
         ├─ community/
         ├─ official/
         ├─ orbit/
         ├─ research/
         └── ...
     ```

5. **Install Protobuf and Compile Protobuf** 
 
    1. Install Protobuf
        ```
          sudo apt install protobuf-compiler
        ```
    2. cd to `Tensorflow/models/research/` directory and run the following command:
    
       ```
        # from within Tensorflow/models/research/
        protoc object_detection/protos/*.proto --python_out=.
       ```
6. **Install COCO API**
     
     Run the following command:
     
     ```
      # in directory of your choice (run the following commands line by line)
      git clone https://github.com/cocodataset/cocoapi.git
      cd cocoapi/PythonAPI
      make
      cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/research/
     ```
7. **Install Object Detection API**
    
    ```
     # from within Tensorflow/models/research/ (run the following commands line by line)
     cp object_detection/packages/tf2/setup.py .
     python -m pip install --use-feature=2020-resolver .
    ```

8. **Test Installation**
   
   ```
    # from within TensorFlow/models/research/
    python object_detection/builders/model_builder_tf2_test.py
   ```
   
   Expected output:
   
   ```
    [       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
    [ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
    INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update): 0.0s
    I0608 18:49:13.183754 29296 test_util.py:2102] time(__main__.ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update): 0.0s
    [       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
    [ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
    INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold): 0.0s
    I0608 18:49:13.186750 29296 test_util.py:2102] time(__main__.ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold): 0.0s
    [       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
    [ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
    INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_model_config_proto): 0.0s
    I0608 18:49:13.188250 29296 test_util.py:2102] time(__main__.ModelBuilderTF2Test.test_invalid_model_config_proto): 0.0s
    ...
   ```
   
## References
