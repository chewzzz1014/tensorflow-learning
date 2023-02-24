## Overview
Using Tensorflow Object Detection API with 10 train images, 4 test images and 4 detection classes.

## Project Structure
```
training_demo/
├─ annotations/
├─ exported-models/
│  └─ my_model/
│     ├─ checkpoint/
│     ├─ saved_model/
│     └─ pipeline.config
├─ images/
│  ├─ test/
│  └─ train/
├─ models/
│  └─ my_ssd_resnet50_v1_fpn/
│     └─ pipeline.config
├─ pre-trained-models/
└─ README.md
```

## Train Result
Training with num_steps: 500

![image](https://user-images.githubusercontent.com/92832451/220566068-f03dd1db-f135-4ad5-bb56-33872764ac7c.png)

![image](https://user-images.githubusercontent.com/92832451/220566156-2ef3a669-361f-48f0-a243-525214d75a17.png)



## References
1. https://colab.research.google.com/github/mlnuggets/maskrcnn/blob/main/Object_detection_with_TensorFlow_2_Object_detection_API.ipynb

2. https://colab.research.google.com/drive/1QCU_dCR0ozI8j6X2btEDCsaUk5p_b1uw?usp=sharing

3. https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/
