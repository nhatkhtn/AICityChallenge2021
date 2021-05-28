""" Sample TensorFlow image-directory-to-TFRecord

usage: generate_tfrecord.py [-h] [-i IMAGE_DIR] [-o OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
"""

import os
import io
import argparse
from pathlib import Path

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
import pandas as pd
from object_detection.utils import dataset_util

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="TensorFlow CSV-to-TFRecord converter")
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. ",
                    type=str)
parser.add_argument("-a",
                    "--annotations",
                    help="Path to annotations csv. ",
                    type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)


args = parser.parse_args()


def create_tf_example(img_path, df):
    # print(str(img_path).replace('\\\\', '\\'))
    annos = df[df.iloc[:, 0] == str(img_path).replace('\\\\', '\\')]

    with tf.gfile.GFile(str(img_path), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = img_path.stem.encode('utf8')
    image_format = b'jpg'

    xmins = annos.iloc[:, 1] / width
    xmaxs = annos.iloc[:, 3] / width
    ymins = annos.iloc[:, 2] / height
    ymaxs = annos.iloc[:, 4] / height
    classes_text = [b'vehicle'] * len(annos)
    classes = [1] * len(annos)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    df = pd.read_csv(args.annotations, header=None)
    writer = tf.python_io.TFRecordWriter(args.output_path)
    img_dir = Path(args.image_dir)
    for img_path in tqdm(img_dir.iterdir()):
        tf_example = create_tf_example(img_path, df)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(args.output_path))


if __name__ == '__main__':
    tf.app.run()
