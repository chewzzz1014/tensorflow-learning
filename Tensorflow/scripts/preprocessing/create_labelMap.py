# adjusted from: https://github.com/datitran/raccoon_dataset
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf


def xml_to_csv(path):
    classes_names = []
    xml_list = []

    print(path)
    f_path = os.path.join(path, '*.xml')

    for xml_file in glob.glob(f_path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            classes_names.append(member[0].text)
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    classes_names = list(set(classes_names))
    print(classes_names)
    # classes_names.sort()
    return xml_df, classes_names


all_classes = []
for label_path in ['train', 'test']:
    image_path = os.path.join(os.path.abspath(
        'C:/Users/USER/tensorflow-learning/Tensorflow/workspace/training_demo/images/'), label_path)
    xml_df, classes = xml_to_csv(image_path)
    for c in classes:
        if c not in all_classes:
            all_classes.append(c)
    xml_df.to_csv(f'{label_path}.csv', index=None)
    print(f'Successfully converted {label_path} xml to csv.')

label_map_path = os.path.join(os.path.abspath(
    'C:/Users/USER/tensorflow-learning/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt'))
pbtxt_content = ""
all_classes.sort()

for i, class_name in enumerate(all_classes):
    pbtxt_content = (
        pbtxt_content
        +
        "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(
            i + 1, class_name)
    )

pbtxt_content = pbtxt_content.strip()

with open(label_map_path, "w") as f:
    f.write(pbtxt_content)
    print('Successfully created label_map.pbtxt ')
