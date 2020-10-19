from styx_msgs.msg import TrafficLight
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
from cv_bridge import CvBridge
import rospy

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model = load_model('/home/workspace/yolo/model.h5')
        self.model._make_predict_function()
        # define the expected input shape for the model - this does not change
        self.input_w, self.input_h = 416, 416

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def decode_netout(self, netout, anchors, obj_thresh, net_h, net_w):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5
        boxes = []
        netout[..., :2] = self._sigmoid(netout[..., :2])
        netout[..., 4:] = self._sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for i in range(grid_h * grid_w):
            row = i / grid_w
            col = i % grid_w
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[int(row)][int(col)][b][4]
                if objectness.all() <= obj_thresh: continue
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]
                x = (col + x) / grid_w  # center position, unit: image width
                y = (row + y) / grid_h  # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
                # last elements are class probabilities
                classes = netout[int(row)][col][b][5:]
                box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                boxes.append(box)
        return boxes

    def correct_yolo_boxes(self, boxes, image_h, image_w, net_h, net_w):
        new_w, new_h = net_w, net_h
        for i in range(len(boxes)):
            x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
            y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
        union = w1 * h1 + w2 * h2 - intersect
        return float(intersect) / union

    def do_nms(self, boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return
        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])
            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                if boxes[index_i].classes[c] == 0: continue
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0


    # get all of the results above a threshold
    def get_boxes(self, boxes, thresh):
        v_boxes, v_scores = list(), list()
        # enumerate all boxes
        for box in boxes:
            # enumerate all possible labels
            if box.classes[9] > thresh: #Looking only for traffic lights - label 10
                v_boxes.append(box)
                v_scores.append(box.classes[9] * 100)
            
        return v_boxes, v_scores

    def draw_boxes(self, filename, image, v_boxes, v_scores):

        # plot each box
        for i in range(len(v_boxes)):
            box = v_boxes[i]
            # get coordinates
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            # calculate width and height of the box
            cv2.rectangle(image,(box.xmin,box.ymin),(box.xmax,box.ymax),(0,255,0),3)
            # save image
            cv2.imwrite(filename,image)
            

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # getting the size of the image       
        image_h,  image_w , channel = image.shape
        # scale pixel values to [0, 1]
        img = cv2.resize(image,(self.input_w, self.input_h))
        img = img.astype('float32')
        img /= 255.0
        # add a dimension so that we have one sample
        img = expand_dims(img, 0)
        # perform prediction
        yhat = self.model.predict(img)
        anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
        # define the probability threshold for detected objects
        class_threshold = 0.8
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += self.decode_netout(yhat[i][0], anchors[i], class_threshold, self.input_h, self.input_w)
        # correct the sizes of the bounding boxes for the shape of the image
        self.correct_yolo_boxes(boxes, image_h, image_w, self.input_h, self.input_w)
        # suppress non-maximal boxes
        self.do_nms(boxes, 0.5)
        # get the details of the traffic light objects
        v_boxes, v_scores = self.get_boxes(boxes, class_threshold)
        #name = '/home/workspace/traffic_2.jpg'
        #self.draw_boxes(name, image, v_boxes, v_scores)
        
        state = 4
        for i in range(len(v_boxes)):
            #crop_img = img[v_boxes[i].miny:v_boxes[i].maxy, v_boxes[i].miny:v_boxes[i].maxy]
            #do something to detect if red
            #return state = 0
            pass
        
        return state
