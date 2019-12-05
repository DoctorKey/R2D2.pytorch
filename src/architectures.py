# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import torch
import torch.nn as nn
import torchvision

from .model.shakeshake import ResNet32x32, ShakeShakeBlock

from .utils import export, parameter_count


"""
    CIFAR-10 Model
"""

@export
def cifar_shakeshake26(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet32x32(ShakeShakeBlock,
                        layers=[4, 4, 4],
                        channels=96,
                        downsample='shift_conv', **kwargs)
    return model


def create_model(model_name, num_classes, detach_para=False):
    model_factory = globals()[model_name]
    model_params = dict(pretrained=False, num_classes=num_classes)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).cuda()

    if detach_para:
        for param in model.parameters():
            param.detach_()
    return model