import logging
import os
import sys

from object_detection.train import main as train_main
from object_detection.utils import label_map_util
import tensorflow as tf

from util import render_config

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# BASE_DIR = r'/home/xdynix/PycharmProjects/' \
#           r'tableware-recognition/tableware-recognition'

BASE_DIR = os.path.dirname(__file__)

flags = tf.app.flags
flags.DEFINE_string('model_dir', os.path.join(BASE_DIR, 'model'),
                    'Path to directory to store models.')
flags.DEFINE_string('data_dir', os.path.join(BASE_DIR, 'data'),
                    'Path to directory to store data.')
flags.DEFINE_string(
    'checkpoint',
    os.path.join(
        BASE_DIR,
        'pre_trained_models',
        'faster_rcnn_resnet101_coco_2018_01_28',
        'model.ckpt',
    ),
    'Path to pre-trained model.',
)
flags.DEFINE_string('run_name', 'faster_rcnn_resnet101', '')
FLAGS = flags.FLAGS

def main(_):
    working_dir = os.path.join(FLAGS.model_dir, FLAGS.run_name)

    os.makedirs(working_dir, exist_ok=True)

    label_map_path = os.path.join(FLAGS.data_dir, 'label_map.pbtxt')
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    with open(os.path.join(FLAGS.data_dir, 'example_num.txt'), 'rt') as f:
        eval_num_examples = f.read().split()[1]

    config_str = render_config(
        os.path.join('config', 'faster_rcnn_resnet101.config'),
        num_classes=len(label_map_dict),
        fine_tune_checkpoint="",
        train_input_path=os.path.abspath(
            os.path.join(FLAGS.data_dir, 'train.record')
        ),
        eval_input_path=os.path.abspath(
            os.path.join(FLAGS.data_dir, 'val.record')
        ),
        label_map_path=os.path.abspath(label_map_path),
        eval_num_examples=eval_num_examples,
        eval_num_visualizations=str(min(int(eval_num_examples), 50)),
    )
    config_path = os.path.abspath(os.path.join(working_dir, 'config.txt'))
    with open(config_path, 'wt', encoding='utf-8') as f:
        f.write(config_str)

    # FLAGS.logtostderr = True
    FLAGS.pipeline_config_path = config_path
    FLAGS.train_dir = os.path.abspath(os.path.join(working_dir, 'train'))
    train_main(_)


if __name__ == '__main__':
    tf.app.run()