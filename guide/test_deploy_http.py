# TODO:
# 1. Move code into method
# 2. Image path and output path passed thru command line args
# 3. Convert to JS / execute using JS

import numpy as np
from PIL import Image
import json
import requests
import tensorflow as tf
import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

image_path = '/root/tensorflow/examples/image1.jpg'
model_url = 'http://167.99.78.252:9000/v1/models/obj_det/versions/1:predict'


def load_image_into_numpy_array(path):
    return np.array(Image.open(path).convert('RGB'))


def send_post_req():
    # data and headers for POST request
    data = json.dumps({
        'signature_name': 'serving_default',
        'instances': input_tensor.numpy().tolist()
        # 'instances': input_tensor.numpy()
    })
    headers = {'content-type': 'application/json'}
    # send POST request
    json_res = requests.post(model_url, data=data, headers=headers)
    predictions = json.loads(json_res.text)
    return predictions


def log_result(predictions, path_name):
    with open(path_name, 'w') as obj:
        json.dump(predictions, obj)


def draw_boxes(predictions, output_image_path):
    # get label map
    category_index = label_map_util.create_category_index_from_labelmap(
        "./workspace/training_demo/annotations/label_map.pbtxt", use_display_name=True)

    # convert classes number to integer
    predictions['predictions'][0]['detection_classes'] = np.array(predictions['predictions'][0]['detection_classes']).astype(
        np.int64)

    # draw boxes
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        np.array(predictions['predictions'][0]['detection_boxes']),
        predictions['predictions'][0]['detection_classes'],
        predictions['predictions'][0]['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        # Adjust this value to set the minimum probability boxes to be classified as True
        min_score_thresh=0.1,
        agnostic_mode=False)
    plt.figure(figsize=(12, 8), dpi=200)
    plt.axis("off")
    plt.imshow(image_np_with_detections)
    plt.show()
    plt.savefig(output_image_path)


# conver image -> np array -> tensor
image_np = load_image_into_numpy_array(image_path)
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]
# for drawing boxes
image_np_with_detections = image_np.copy()

# send post reqeust
predictions = send_post_req()
# log result into file
log_result(predictions, './result_http.txt')
# draw boxes
draw_boxes(predictions, './result_http.jpg')
