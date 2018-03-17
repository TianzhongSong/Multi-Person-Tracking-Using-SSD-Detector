# -*- coding: utf-8 -*-
from HOG import hog
import numpy as np


def Extract_feature(input_img):
    feature = hog(input_img, all_norms=True)
    feature = np.concatenate((feature[0], feature[1], feature[2], feature[3]), axis=0)
    return feature
