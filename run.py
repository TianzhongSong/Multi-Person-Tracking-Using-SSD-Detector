# -*- coding:utf-8 -*-
"""
Created on Feb, 23, 2018
@author: Tianz
"""
from time import sleep
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
from ssd_utils import BBoxUtility
from ssd import SSD300 as SSD
from tracker import Tracker
from get_features import *


def run_camera(input_shape, model):
    num_classes = 21
    conf_thresh = 0.5
    bbox_util = BBoxUtility(num_classes)
    vid = cv2.VideoCapture(0)
    sleep(1.0)
    # Compute aspect ratio of video
    vidw = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    vidh = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    trackers = Tracker()
    while True:
        ret, origin_image = vid.read()
        frame = origin_image
        if not ret:
            print("Done!")
            return None
        im_size = (input_shape[0], input_shape[1])
        resized = cv2.resize(frame, im_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        inputs = [image.img_to_array(rgb)]
        tmp_inp = np.array(inputs)
        x = preprocess_input(tmp_inp)
        y = model.predict(x)
        results = bbox_util.detection_out(y)
        if len(results) > 0 and len(results[0]) > 0:
            det_label = results[0][:, 0]
            det_conf = results[0][:, 1]
            det_xmin = results[0][:, 2]
            det_ymin = results[0][:, 3]
            det_xmax = results[0][:, 4]
            det_ymax = results[0][:, 5]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            if 15 not in top_label_indices:
                pass
            else:
                trackers.bbox = []
                trackers.features_current = []
                trackers.index = []
                for i in range(top_conf.shape[0]):
                    class_num = int(top_label_indices[i])
                    if class_num == 15:
                        xmin = int(round((top_xmin[i] * vidw) * 0.9))
                        ymin = int(round((top_ymin[i] * vidh) * 0.9))
                        xmax = int(round((top_xmax[i] * vidw) * 1.1)) if int(round(
                            (top_xmax[i] * vidw)) * 1.1) <= vidw else int(round(
                            top_xmax[i] * vidw))
                        ymax = int(round((top_ymax[i] * vidh) * 1.1)) if int(round(
                            (top_ymax[i] * vidh) * 1.1)) <= vidh else int(round(top_ymax[i] * vidh))
                        curWindow = [xmin, ymin, xmax, ymax]
                        trackers.bbox.append(curWindow)
                        feature = Extract_feature(frame[ymin:ymax, xmin:xmax, :])
                        trackers.features_current.append(feature)
                if trackers.features_previous is None:
                    trackers.index.append(i for i in range(len(trackers.bbox)))
                    for item, index in trackers.bbox, trackers.index:
                        cv2.rectangle(frame, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (255, 0, 0), 2)
                        cv2.putText(frame, "person: {}".format(index + 1), (item[0] + 10, item[1] + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                else:
                    trackers.match()
                    trackers.update()
                    for item, index in trackers.bbox, trackers.index:
                        cv2.rectangle(frame, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (255, 0, 0), 2)
                        cv2.putText(frame, "person: {}".format(index + 1), (item[0] + 10, item[1] + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.imshow('tracking', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    input_shape = (300, 300, 3)
    ssd_model = SSD(input_shape, num_classes=21)
    ssd_model.load_weights('weights_SSD300.hdf5')
    run_camera(input_shape, ssd_model)
