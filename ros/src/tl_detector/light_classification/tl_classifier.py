from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import tensorflow as tf


class TLClassifier(object):
    def __init__(self, model_path=None):
        

        self.graph = self.load_graph(model_path)
        self.sess = tf.Session(graph=self.graph)

    def load_graph(self, path):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')
        
        return graph
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        
        img_numpy = np.asarray(image[:,:])
        img_expd = np.expand_dims(img_numpy, axis=0)
        with self.graph.as_default():
            
            img_tns = self.graph.get_tensor_by_name('image_tensor:0')
            det_bx = self.graph.get_tensor_by_name('detection_boxes:0')
            det_scr = self.graph.get_tensor_by_name('detection_scores:0')
            det_cls = self.graph.get_tensor_by_name('detection_classes:0')
            num_det = self.graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num) = self.sess.run([det_bx, det_scr, det_cls, num_det],feed_dict={img_tns: img_expd})
            
            if classes[0][0]==1:
                return TrafficLight.GREEN
            if classes[0][0]==2:
                return TrafficLight.RED
            if classes[0][0]==3:
                return TrafficLight.YELLOW
            if classes[0][0]==4:
                return TrafficLight.UNKNOWN
                                                   
        
        return TrafficLight.UNKNOWN


   