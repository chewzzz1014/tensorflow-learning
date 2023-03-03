# Object Detection Model Training Using TensorFlow 2 Object Detection API

## Pre
   - Make sure that the following have been satisfied:
   
     ```
     1. Tensorflow, TensorFlow Object Detection API, Protobuf, COCO API, LabelImg have been installed
     2. All images along with their annotations (in xml format) are stored under workspace/training_demo/images (No need to split them into train adn test first)
     3. Has the following project directory:
         
         TensorFlow/
          ├─ addons/ (Optional)
          │  └─ labelImg/
          └─ models/
             ├─ community/
             ├─ official/
             ├─ orbit/
             ├─ research/
             └─ ...
          ├─ scripts/
          │  └─ preprocessing/
          └─ workspace/
             └─ training_demo/
                ├─ annotations/
                ├─ exported-models/
                ├─ images/
                ├─ models/
                ├─ pre-trained-models/
                └─ README.md
     ```
     
## Steps
1. Partion Dataset into Train and Test
   - Download the `partition_dataset.py` script from [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) into `scripts/preprocessing'
   - cd to `scripts/preprocessing' and run the following command:
   - 
     ```
      python partition_dataset.py -x -i [PATH_TO_IMAGES_FOLDER] -r 0.1
      # For example
      # python partition_dataset.py -x -i C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/images -r 0.1
     ````
     
2. Create Label Map
   - Label Map will contain all classes name
   - Download the `partition_dataset.py` script from [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) into `scripts/preprocessing'
   - cd to `scripts/preprocessing` and run the following command: 
   
     ```
        # Create train data:
         python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record
        
        # Create test data:
        python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record

        # For example
        # python generate_tfrecord.py -x C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/images/train -l C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt -o C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/annotations/train.record
        # python generate_tfrecord.py -x C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/images/test -l C:/Users/sglvladi/Documents/Tensorflow2/workspace/training_demo/annotations/label_map.pbtxt -o C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/annotations/test.record
     ```
     
 3. Download Pre-Trained Model
    - Choose one model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
    - Download the model under the `pre-trained-models` directory:

      ```
          training_demo/
          ├─ ...
          ├─ pre-trained-models/
          │  └─ ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/
          │     ├─ checkpoint/
          │     ├─ saved_model/
          │     └─ pipeline.config
          └─ ...
      ```
      
 4. Configure Training Pipeline
    - Create a new directory, `my_[model_name]` under `training_demo/models`
    - Copy `pipeline.config` from `pre-trained-models/[model_name]` into  `training_demo/models/my_[model_name]`
    
      ```
        training_demo/
        ├─ ...
        ├─ models/
        │  └─ my_ssd_resnet50_v1_fpn/
        │     └─ pipeline.config
        └─ ...
      ```
     - Edit params in pipeline.config

5. Train Model
   - cd to `training_demo` and run the following command:
      
      ```
         python model_main_tf2.py --model_dir=models/my_[model_name] --pipeline_config_path=models/my_[model_name]/pipeline.config
      
         # for example
        python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config
      ```
      
      
6. Export Trained Model
   - Copy the `/models/research/object_detection/exporter_main_v2.py` into `training_demo`
   - cd to `trainig_demo` and run the following command:

     ```
       python ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_[model_name]/pipeline.config --trained_checkpoint_dir ./models/my_[model_name]/ --output_directory ./exported-models/my_model
       
       # for example
       python ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_ssd_resnet50_v1_fpn/pipeline.config --trained_checkpoint_dir ./models/my_ssd_resnet50_v1_fpn/ --output_directory ./exported-models/my_model
     ```
     
   - Location of Saved Model:

     ```
       training_demo/
        ├─ ...
        ├─ exported-models/
        │  └─ my_model/
        │     └─ saved_model/
        |         ├─ assets/
        |         ├─ variables/
        |         └─saved_model.pb
        |     ├─ checkpoint/
        │     └─ pipeline.config
        └─ ...
     ```
     
  7. Move save_model directory under version number
  
      ```
       training_demo/
        ├─ ...
        ├─ exported-models/
        │  └─ my_model/
        │     └─ saved_model/
        │         └─ 1/
        |             ├─ assets/
        |             ├─ variables/
        |             └─saved_model.pb
        |     ├─ checkpoint/
        │     └─ pipeline.config
        └─ ...
     ```
