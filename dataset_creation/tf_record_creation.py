# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw COCO dataset to TFRecord for object_detection.

This tool supports data generation for object detection (boxes, masks),
keypoint detection, and DensePose.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import tensorflow_io as tfio

import hashlib
import io
import json
import logging
import os
import contextlib2
import numpy as np
import PIL.Image

from pycocotools import mask
# import tensorflow  as tf
import tensorflow.compat.v1  as tf
# tensorflow.compat.v1
# from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
tf.flags.DEFINE_boolean(
    'include_masks', False, 'Whether to include instance segmentations masks '
    '(PNG encoded) in the result. default: False.')
tf.flags.DEFINE_string('train_image_dir', '', 'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '', 'Validation image directory.')
tf.flags.DEFINE_string('test_image_dir', '', 'Test image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('testdev_annotations_file', '',
                       'Test-dev annotations JSON file.')
tf.flags.DEFINE_string('train_keypoint_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_keypoint_annotations_file', '',
                       'Validation annotations JSON file.')
# DensePose is only available for coco 2014.
tf.flags.DEFINE_string('train_densepose_annotations_file', '',
                       'Training annotations JSON file for DensePose.')
tf.flags.DEFINE_string('val_densepose_annotations_file', '',
                       'Validation annotations JSON file for DensePose.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
# Whether to only produce images/annotations on person class (for keypoint /
# densepose task).
tf.flags.DEFINE_boolean('remove_non_person_annotations', False, 'Whether to '
                        'remove all annotations for non-person objects.')
tf.flags.DEFINE_boolean('remove_non_person_images', False, 'Whether to '
                        'remove all examples that do not contain a person.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

_COCO_KEYPOINT_NAMES = [
    b'nose', b'left_eye', b'right_eye', b'left_ear', b'right_ear',
    b'left_shoulder', b'right_shoulder', b'left_elbow', b'right_elbow',
    b'left_wrist', b'right_wrist', b'left_hip', b'right_hip',
    b'left_knee', b'right_knee', b'left_ankle', b'right_ankle'
]

_COCO_PART_NAMES = [
    b'torso_back', b'torso_front', b'right_hand', b'left_hand', b'left_foot',
    b'right_foot', b'right_upper_leg_back', b'left_upper_leg_back',
    b'right_upper_leg_front', b'left_upper_leg_front', b'right_lower_leg_back',
    b'left_lower_leg_back', b'right_lower_leg_front', b'left_lower_leg_front',
    b'left_upper_arm_back', b'right_upper_arm_back', b'left_upper_arm_front',
    b'right_upper_arm_front', b'left_lower_arm_back', b'right_lower_arm_back',
    b'left_lower_arm_front', b'right_lower_arm_front', b'right_face',
    b'left_face',
]

_DP_PART_ID_OFFSET = 1


def clip_to_unit(x):
  return min(max(x, 0.0), 1.0)

def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
      u'width', u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
        u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
        coordinates in the official COCO dataset are given as [x, y, width,
        height] tuples using absolute coordinates where x, y represent the
        top-left (0-indexed) corner.  This function converts to the format
        expected by the Tensorflow Object Detection API (which is which is
        [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
        size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed by the
      'id' field of each category.  See the label_map_util.create_category_index
      function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    keypoint_annotations_dict: A dictionary that maps from annotation_id to a
      dictionary with keys: [u'keypoints', u'num_keypoints'] represeting the
      keypoint information for this person object annotation. If None, then
      no keypoint annotations will be populated.
    densepose_annotations_dict: A dictionary that maps from annotation_id to a
      dictionary with keys: [u'dp_I', u'dp_x', u'dp_y', 'dp_U', 'dp_V']
      representing part surface coordinates. For more information see
      http://densepose.org/.
    remove_non_person_annotations: Whether to remove any annotations that are
      not the "person" class.
    remove_non_person_images: Whether to remove any images that do not contain
      at least one "person" annotation.

  Returns:
    key: SHA256 hash of the image.
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.
    num_keypoint_annotation_skipped: Number of keypoint annotations that were
      skipped.
    num_densepose_annotation_skipped: Number of DensePose annotations that were
      skipped.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  print(image )
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  is_crowd = []
  category_names = []
  category_ids = []
  area = []
  num_annotations_skipped = 0
  num_densepose_annotation_used = 0
  num_densepose_annotation_skipped = 0
  for object_annotations in annotations_list:
    (x, y, width, height) = tuple(object_annotations['bbox'])
    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    category_id = int(object_annotations['category_id'])
    category_name = category_index[category_id]['name'].encode('utf8')
    
    xmin.append(float(x) / image_width)
    xmax.append(float(x + width) / image_width)
    ymin.append(float(y) / image_height)
    ymax.append(float(y + height) / image_height)
    is_crowd.append(object_annotations['iscrowd'])
    category_ids.append(category_id)
    category_names.append(category_name)
    area.append(object_annotations['area'])
  
  image_format = b'jpg'
  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          # dataset_util.bytes_feature('jpeg'.encode('utf8')),
          dataset_util.bytes_feature(image_format),

      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/text':
          dataset_util.bytes_list_feature(category_names),
      'image/class/label':
          dataset_util.int64_list_feature(category_ids),
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return (key, example, num_annotations_skipped)

def _create_tf_record_from_coco_annotations(annotations_file, image_dir,
                                            output_path,
                                            num_shards):
  with contextlib2.ExitStack() as tf_record_close_stack, \
      tf.io.gfile.GFile(annotations_file, 'r') as fid:
    output_tfrecords = tf.io.TFRecordWriter(output_path)
    groundtruth_data = json.load(fid)
    images = groundtruth_data['images']
    category_index = label_map_util.create_category_index(
        groundtruth_data['categories'])

    annotations_index = {}
    if 'annotations' in groundtruth_data:
      logging.info('Found groundtruth annotations. Building annotations index.')
      for annotation in groundtruth_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_index:
          annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)
    missing_annotation_count = 0
    for image in images:
      image_id = image['id']
      if image_id not in annotations_index:
        missing_annotation_count += 1
        annotations_index[image_id] = []
    logging.info('%d images are missing annotations.',
                 missing_annotation_count)
    total_num_annotations_skipped = 0
    for idx, image in enumerate(images):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(images))
      annotations_list = annotations_index[image['id']]
      keypoint_annotations_dict = None
      (_, tf_example, num_annotations_skipped,) = create_tf_example(
           image, annotations_list, image_dir, category_index )
      total_num_annotations_skipped += num_annotations_skipped
      shard_idx = idx % num_shards
      if tf_example:
          output_tfrecords.write(tf_example.SerializeToString())
      logging.info('Finished writing, skipped %d annotations.',
                  total_num_annotations_skipped)
def main(_):
  output_dir= "/media/akilesh/unittest_script/new_dataset/tf/validation/train11/"
  train_image_dir="/media/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/"
  # val_image_dir ="/media/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/"
  # test_image_dir ="/media/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/"
  train_annotations_file="/media/MIVisionX-data/rocal_data/coco/coco_10_img/annotations/instances_train2017.json"
  if not tf.io.gfile.isdir(output_dir):
    tf.gfile.MakeDirs(output_dir)
  train_output_path = os.path.join(output_dir, 'coco_train.record')

  _create_tf_record_from_coco_annotations(
      train_annotations_file,
      train_image_dir,
      train_output_path,
      num_shards=10)


if __name__ == '__main__':
  tf.compat.v1.app.run()
