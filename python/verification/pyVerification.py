#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-02-22 14:56:11
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-02-22 16:37:40

import caffe
import numpy as np

class VerificationLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to verify.")

    def reshape(self, bottom, top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimensions.")
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data == bottom[1].data
        print top[0].data

    def backward(self, top, propagate_down, bottom):
        pass