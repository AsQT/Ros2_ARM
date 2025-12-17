#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D 
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

TEMPLATE_PATH      = 'Book_2.jpg' 
ORBBEC_IMAGE_TOPIC = '/camera/rgb/image_raw'

class SIFT_YOLO_TrackerNode(Node):
    def __init__(self):
        super().__init__('sift_yolo_tracker_node')
        
        self.bridge           = CvBridge()
        self.get_logger().info(f'Đang nhan ảnh từ topic: {ORBBEC_IMAGE_TOPIC}')
        self.subscription_img = self.create_subscription(
            Image,
            ORBBEC_IMAGE_TOPIC,  
            self.image_callback,
            10
        )
        self.subscription_yolo = self.create_subscription(
            Pose2D, 
            '/yolo_detections',
            self.yolo_detection_callback,
            10
        )
        self.publisher_ = self.create_publisher(Image, '/tracking_result', 10)
        self.sift       = cv2.SIFT_create(nfeatures   =2000)
        self.bf         = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.trajectory = []
        
        self.is_tracking_initialized = False
        self.last_frame              = None
        self.current_frame           = None
        
        self.des_template = None
        self.kp_template  = None
        self.w_temp       = 0
        self.h_temp       = 0
        
        self.initialize_sift_with_default_template()
        
        self.get_logger().info('SIFT-YOLO Tracker Node đã khởi động.')

    def initialize_sift_with_default_template(self):
        try:
            img_template = cv2.imread(TEMPLATE_PATH)
            if img_template is None:
                return

            gray_template                       = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
            self.kp_template, self.des_template = self.sift.detectAndCompute(gray_template, None)
            self.h_temp, self.w_temp            = gray_template.shape
            
            if self.des_template is not None and len(self.des_template) > 10:
                self.is_tracking_initialized = True
            
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def yolo_detection_callback(self, msg):
        if self.current_frame is None:
            return

        x = int(msg.x)
        y = int(msg.y)
        w = int(msg.theta)
        h = w 

        x = max(0, x)
        y = max(0, y)
        w = min(self.current_frame.shape[1] - x, w)
        h = min(self.current_frame.shape[0] - y, h)

        if w > 10 and h > 10:
            roi      = self.current_frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            self.kp_template, self.des_template = self.sift.detectAndCompute(gray_roi, None)
            self.w_temp, self.h_temp            = roi.shape[1], roi.shape[0]
            
            if self.des_template is not None and len(self.des_template) > 10:
                self.is_tracking_initialized    = True
                self.trajectory.clear()

    def image_callback(self, msg):
        try:
            frame              = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_frame = frame.copy() 
            
            if not self.is_tracking_initialized or self.des_template is None:
                cv2.putText(frame, "CHO PHAT HIEN YOLO...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.publish_result(frame)
                return

            gray_frame          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame          = cv2.equalizeHist(gray_frame)
            kp_frame, des_frame = self.sift.detectAndCompute(gray_frame, None)

            if des_frame is None or len(des_frame) < 10:
                cv2.putText(frame, "KHONG DU KEYPOINT SIFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.publish_result(frame)
                return

            matches = self.bf.match(self.des_template, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)[:70]

            if len(matches) > 10: 
                src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    pts = np.float32([[0, 0], [self.w_temp, 0],
                                      [self.w_temp, self.h_temp],
                                      [0, self.h_temp]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3) 
                    
                    center = np.mean(dst, axis=0)[0]
                    self.trajectory.append(center)
                    
                    for i in range(1, len(self.trajectory)):
                        pt1 = tuple(np.int32(self.trajectory[i-1]))
                        pt2 = tuple(np.int32(self.trajectory[i]))
                        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
                    
                    if len(self.trajectory) > 20: 
                        self.trajectory = self.trajectory[-20:]
                else:
                    self.is_tracking_initialized = False 
            else:
                self.is_tracking_initialized = False 
            
            self.publish_result(frame)
            
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def publish_result(self, frame):
        tracking_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher_.publish(tracking_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SIFT_YOLO_TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()