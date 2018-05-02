"""Contains common utility functions."""
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import distutils.util
import numpy as np
from paddle.fluid import core
import paddle.fluid.layers.tensor as tensor
import paddle.fluid.layers.nn as nn
import paddle.fluid.layers.control_flow as control_flow
from paddle.fluid.initializer import init_on_cpu

def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).iteritems()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def _decay_step_counter(begin=0):
    # the first global step is zero in learning rate decay
    global_step = nn.autoincreased_step_counter(
        counter_name='@LR_DECAY_COUNTER@', begin=begin, step=1)
    global_step = tensor.cast(global_step, 'float32')
    return global_step

def exponential_decay(learning_rate, decay_steps, decay_rate, staircase=False):
    global_step = _decay_step_counter()
    with init_on_cpu():
        # update learning_rate
        div_res = global_step / decay_steps
        if staircase:
            div_res = ops.floor(div_res)
        decayed_lr = learning_rate * (decay_rate**div_res)
    return decayed_lr

def piecewise_decay_with_warmup(boundaries, values):
    if len(values) - len(boundaries) != 1:
        raise ValueError("len(values) - len(boundaries) should be 1")

    WARM_UP_ITERS = 500.0
    WARM_UP_FACTOR = 1.0 / 3.0

    global_step = _decay_step_counter()
    with init_on_cpu():
        lr = tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate")
        with control_flow.Switch() as switch:
            with switch.case(global_step < WARM_UP_ITERS):
                alpha = global_step / WARM_UP_ITERS
                warmup_factor = WARM_UP_FACTOR * (1 - alpha) + alpha
                warmup_val = (values[0] * warmup_factor)
                tensor.assign(warmup_val, lr)
            for i in range(len(boundaries)):
                boundary_val = tensor.fill_constant(
                    shape=[1], dtype='float32', value=float(boundaries[i]))
                value_var = tensor.fill_constant(
                    shape=[1], dtype='float32', value=float(values[i]))
                with switch.case(global_step < boundary_val):
                    tensor.assign(value_var, lr)
            last_value_var = tensor.fill_constant(
                shape=[1],
                dtype='float32',
                value=float(values[len(values) - 1]))
            with switch.default():
                tensor.assign(last_value_var, lr)
    return lr