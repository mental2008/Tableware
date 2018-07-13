import logging
import os
import re
import sys

from object_detection.export_inference_graph import main as export_main
import tensorflow as tf

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# BASE_DIR = r'/home/xdynix/PycharmProjects/' \
#            r'tableware-recognition/tableware-recognition'
BASE_DIR = os.path.dirname(__file__)

flags = tf.app.flags
flags.DEFINE_string('model_dir', os.path.join(BASE_DIR, 'model'),
                    'Path to directory to store models.')
flags.DEFINE_string('data_dir', os.path.join(BASE_DIR, 'data'),
                    'Path to directory to store data.')
flags.DEFINE_string('run_name', 'faster_rcnn_resnet101', '')

FLAGS = flags.FLAGS

working_dir = os.path.join(BASE_DIR, 'model', 'faster_rcnn_resnet101')
config_path = os.path.join(BASE_DIR, 'model', 'faster_rcnn_resnet101', 'config.txt')

FLAGS.pipeline_config_path = config_path

with open(
        os.path.join(working_dir, 'train', 'checkpoint'),
        'rt', encoding='utf-8',
) as f:
    contents = f.read()
    checkpoint_path = re.search(
        'model_checkpoint_path: "(.*)"',
        contents,
    ).group(1)

FLAGS.input_type = 'image_tensor'
FLAGS.trained_checkpoint_prefix = checkpoint_path
FLAGS.output_directory = os.path.join('inference_graph', 'faster_rcnn_resnet101')

def main(_):

    # working_dir = os.path.join(FLAGS.model_dir, FLAGS.run_name)

    export_main(_)


if __name__ == '__main__':
    tf.app.run()