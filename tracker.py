# -*- coding: utf-8 -*-
import numpy as np
import copy


class Tracker(object):
    def __init__(self):
        self.bbox = []
        self.index = []
        self.features_previous = []
        self.features_current = []

    def update(self):
        self.features_previous = copy.deepcopy(self.features_current)

    def match(self):
        # 待更新
        pass
