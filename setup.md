## Setup (Following docs)
1. Create and activate virtual environment (using conda in this case)
```
conda create -n [dir name] pip python=3.9
conda activate [dir name]
```

2. Install Tensorflow
```
pip install --ignore-installed --upgrade tensorflow==2.5.0
# verify installation
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

3. Install TensorFlow Object Detection API 
```
#Create a new folder under a path of your choice and name it TensorFlow. (e.g. C:\Users\sglvladi\Documents\TensorFlow).
# cd into the TensorFlow directory.
# download model using git clone/zip files from  the [TensorFlow Models repository](https://github.com/tensorflow/models)
```

```
TensorFlow/
└─ models/
   ├─ community/
   ├─ official/
   ├─ orbit/
   ├─ research/
   └── ...
```

4. Protobuf Installation/Compilation
    1. Head to the [protoc releases page](https://github.com/google/protobuf/releases)

    2. Download the latest protoc-*-*.zip release (e.g. protoc-3.12.3-win64.zip for 64-bit Windows)

    3. Extract the contents of the downloaded protoc-*-*.zip in a directory <PATH_TO_PB> of your choice (e.g. C:\Program Files\Google Protobuf)

    4. Add <PATH_TO_PB>\bin to your Path environment variable (see Environment Setup)

    5. In a new Terminal 1, cd into TensorFlow/models/research/ directory and run the following command:
        ```
        # From within TensorFlow/models/research/
        protoc object_detection/protos/*.proto --python_out=.
        ```

5. COCO API installation
   - Visual C++ 2015 build tools must be installed and on the path
```
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

6. Install the Object Detection API

```
# From within TensorFlow/models/research/
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

```
# test the installation
# From within TensorFlow/models/research/
python object_detection/builders/model_builder_tf2_test.py
```

## Setup (Following Referenced Notebook)
1. Install TensorFlow 2 Object Detection API from Github (in root dir)

```
# clone repo 
git clone https://github.com/tensorflow/models.git

# delete nested .git 
cd models
rm .git -r

# Compile protos.
cd research
protoc object_detection/protos/*.proto --python_out=.

# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

2. Download Mask RCNN model (in root dir)
```
# library for downloading
pip install wget 

# download and unzip model
import wget
model_link = "http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"
wget.download(model_link)

import tarfile
tar = tarfile.open('/content/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz')
tar.extractall('.') 
tar.close()
```

3. Install opencv

```
pip uninstall opencv-python-headless==4.5.5.62
pip install opencv-python-headless==4.5.2.52
```


4. Install libcudnn8 (in bash)

```
apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
```
