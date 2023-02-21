## Setup
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

