import os
import time
import numpy as np
import argparse
import functools
import shutil

import paddle
import paddle.fluid as fluid
import reader
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description="debug nan")
parser.add_argument(
    "--param_path",
    type=str,
    default='train_voc_wiki/0',
    help="The path to params")

args = parser.parse_args()


def freeze2param(image_shape, num_classes, pretrained_model, use_gpu=True):
    devices = os.getenv('CUDA_VISIBLE_DEVICES') or ""
    devices_num = len(devices.split(','))
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')

    locs, confs, box, box_var = mobile_net(num_classes, image, image_shape)
    nmsed_out = fluid.layers.detection_output(
        locs, confs, box, box_var, nms_threshold=0.45)
    #locs, confs, box, box_var, nms_threshold=0.45)
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    scope = fluid.core.Scope()

    def if_exist(var):
        return os.path.exists(os.path.join(pretrained_model, var.name))

    with fluid.scope_guard(scope):
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)
        #numarray_2 = fluid.executor.fetch_var('batch_norm_34.w_2', scope, True)
        #print(numarray_2)
        for i in range(34, 35):
            numarray_0 = fluid.executor.fetch_var('batch_norm_{}.b_0'.format(i),
                                                  scope, True)
            numarray_1 = fluid.executor.fetch_var('batch_norm_{}.w_0'.format(i),
                                                  scope, True)
            numarray_2 = fluid.executor.fetch_var('batch_norm_{}.w_1'.format(i),
                                                  scope, True)
            numarray_3 = fluid.executor.fetch_var('batch_norm_{}.w_2'.format(i),
                                                  scope, True)
            print("======================batch_norm_{}==================".
                  format(i))
            #print(numarray_0)
            #print(numarray_1)
            #print(numarray_2)
            print(numarray_3)
            if np.any(np.isnan(numarray_3)):
                print(np.isnan(numarray_3))

    for i in range(34, 34):
        #numarray_0 = fluid.executor.fetch_var('conv2d_{}.b_0'.format(i), scope, True)
        numarray_1 = fluid.executor.fetch_var('conv2d_{}.w_0'.format(i), scope,
                                              True)
        print("======================conv2d_{}==================".format(i))
        #print(numarray_0)
        print(numarray_1)

    fluid.io.save_inference_model(
        "model/freeze_model_param", ['image'],
        nmsed_out,
        exe,
        main_program=fluid.default_main_program(),
        model_filename='train_wiki_model',
        params_filename='train_wiki_params')


if __name__ == '__main__':
    freeze2param(
        image_shape=[3, 300, 300],
        num_classes=91,
        pretrained_model=args.param_path)
