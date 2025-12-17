#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D, PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import tf2_ros
import tf2_geometry_msgs
import numpy as np

class YoloSmartNode(Node):
    def __init__(self):
        super().__init__('yolo_publisher_node')
        
        self.model       = YOLO('/home/asus/ros2_arm_ws/src/vision_tracking/resource/wood.pt')
        self.bridge      = CvBridge()
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.fx = None; self.cx = None; self.fy = None; self.cy = None
        
        self.CAMERA_FRAME_ID  = "astra_link_optical" 
        self.TABLE_Z = 1.02
        self.latest_depth_img = None

        self.create_subscription(CameraInfo, '/astra/rgb/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/astra/rgb/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, '/astra/depth/image_raw', self.depth_callback, 10)
        
        self.pub_debug = self.create_publisher(Image, '/yolo_debug', 10)
        self.pub_data  = self.create_publisher(Pose2D, '/yolo_coord', 10)

        self.get_logger().info("--- TOA DO TAI: ---")

    def info_callback(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]; self.cx = msg.k[2]; self.fy = msg.k[4]; self.cy = msg.k[5]

    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except: pass

    def rgb_callback(self, msg):
        if self.fx is None or self.latest_depth_img is None: return

        frame      = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results    = self.model.predict(frame, conf=0.8, verbose=False)       
        depth_copy = self.latest_depth_img.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            h, w, _        = frame.shape
            x1, y1         = max(0, x1), max(0, y1)
            x2, y2         = min(w, x2), min(h, y2)

            u_center = (x1 + x2) // 2
            v_center = (y1 + y2) // 2
            
            Z_final = 1.0
            roi_depth = depth_copy[y1:y2, x1:x2]
            mask      = (roi_depth > 0.1) & (roi_depth < (self.TABLE_Z - 0.02))

            if np.any(mask):
                Z_final = np.min(roi_depth[mask])
            else:
                try:
                    z_center = depth_copy[v_center, u_center]
                    if z_center > 0.1: Z_final = z_center
                except: pass

            Xc = (u_center - self.cx) * Z_final / self.fx
            Yc = (v_center - self.cy) * Z_final / self.fy
            
            point_base = self.get_tf_transform(Xc, Yc, Z_final)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (u_center, v_center), 1, (0, 0, 255), -2) 
                       
            if point_base:
                Xb, Yb = point_base.x, point_base.y
            
                text_base = f"BASE: {Xb:.2f}, {Yb:.2f}"
                text_cam  = f"CAM : {Xc:.2f}, {Yc:.2f}" 
                
                cv2.putText(frame, text_base, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, text_cam, (x1, y2 + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                pose_msg               = Pose2D()
                pose_msg.x, pose_msg.y = Xb, Yb
                pose_msg.theta         = float(x2 - x1) 
                self.pub_data.publish(pose_msg)
                
                print(f"Object Center: Base({Xb:.2f}, {Yb:.2f})")

        cv2.imshow("YOLO DETECTION", frame)
        cv2.waitKey(1)
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    def get_tf_transform(self, x, y, z):
        p                               = PointStamped()
        p.header.frame_id               = self.CAMERA_FRAME_ID
        p.header.stamp                  = self.get_clock().now().to_msg()
        p.point.x, p.point.y, p.point.z = float(x), float(y), float(z)
        try:
            if self.tf_buffer.can_transform("base_link", self.CAMERA_FRAME_ID, rclpy.time.Time()):
                return self.tf_buffer.transform(p, "base_link").point
        except: pass
        return None

def main():
    rclpy.init()
    node = YoloSmartNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()