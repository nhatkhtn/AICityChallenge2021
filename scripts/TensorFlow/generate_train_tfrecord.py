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

cnt_annos = 0

def create_tf_example(img_path, annos):
    with tf.gfile.GFile(str(img_path), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    assert width == 800 and height == 410

    filename = img_path.stem.encode('utf8')
    image_format = b'jpg'

    global cnt_annos

    features = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
    }

    if annos is not None:
        xmins = annos.iloc[:, 1] / width
        xmaxs = annos.iloc[:, 3] / width
        ymins = annos.iloc[:, 2] / height
        ymaxs = annos.iloc[:, 4] / height
        classes_text = [b'vehicle'] * len(annos)
        classes = [1] * len(annos)

        cnt_annos += len(annos)

        if annos.empty:
            print(f"Warning: {filename} has no annotations.")

        features.update({
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        })

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example


def main(_):
    img_dir = Path(args.image_dir)
    if args.annotations:
        annos_df = pd.read_csv(args.annotations, header=None)
        print("Generating train/val records.")
    else:
        annos_df = None
        print("Generating test reccords.")

    with tf.python_io.TFRecordWriter(args.output_path) as writer:
        cnt_images = 0
        for img_path in img_dir.iterdir():
            cnt_images += 1
            img_name = img_path.name

            annos = annos_df[annos_df[0] == img_name] if annos_df is not None else None
            tf_example = create_tf_example(img_path, annos)
            writer.write(tf_example.SerializeToString())

        print(f"Number of images in split: {cnt_images}")
        print(f"Total number of annotations written: {cnt_annos}")
        print('Successfully created the TFRecord file: {}'.format(args.output_path))


if __name__ == '__main__':
    tf.app.run()
