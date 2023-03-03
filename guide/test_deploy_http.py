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


image_np = load_image_into_numpy_array(image_path)
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]


data = json.dumps({
    'signature_name': 'serving_default',
    'instances': input_tensor.numpy().tolist()
    # 'instances': input_tensor.numpy()
})
headers = {'content-type': 'application/json'}

# print(test_image)
# print(test_image.shape)

json_res = requests.post(model_url, data=data, headers=headers)
predictions = json.loads(json_res.text)

with open('./result_http.txt', 'w') as obj:
    json.dump(predictions, obj)


# draw boxes
image_np_with_detections = image_np.copy()

category_index = label_map_util.create_category_index_from_labelmap(
    "./workspace/training_demo/annotations/label_map.pbtxt", use_display_name=True)


print(predictions['predictions'][0])
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    predictions['predictions'][0]['detection_boxes'],
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
plt.savefig('./result_' + 'pred_http' + '.jpg')
