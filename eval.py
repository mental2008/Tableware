import logging
import os
import sys

from object_detection.eval import main as eval_main
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


def main(_):
    working_dir = os.path.join(FLAGS.model_dir, FLAGS.run_name)

    config_path = os.path.abspath(os.path.join(working_dir, 'config.txt'))

    # FLAGS.logtostderr = True
    FLAGS.pipeline_config_path = config_path
    FLAGS.checkpoint_dir = os.path.abspath(os.path.join(working_dir, 'train'))
    FLAGS.eval_dir = os.path.abspath(os.path.join(working_dir, 'eval'))
    eval_main(_)


if __name__ == '__main__':
    tf.app.run()
