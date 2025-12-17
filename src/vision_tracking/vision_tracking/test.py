#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import time
import os

MODEL_PATH = '/home/asus/ros2_arm_ws/src/vision_tracking/resource/wood.pt'
TEST_IMAGE_PATH = '/home/asus/ros2_arm_ws/src/vision_tracking/resource/test.jpg'

class YOLOTestNode(Node):
    def __init__(self):
        super().__init__('yolo_test_node')
        
        try:
            self.model = YOLO(MODEL_PATH)
            self.get_logger().info(f'Da load model: {MODEL_PATH}')
        except Exception as e:
            self.get_logger().error(f'LOI LOAD MODEL: {e}')
            return

        if os.path.exists(TEST_IMAGE_PATH):
            self.test_img = cv2.imread(TEST_IMAGE_PATH)
        else:
            self.get_logger().error(f'KHONG TIM THAY FILE ANH: {TEST_IMAGE_PATH}')
            self.test_img = None

        self.bridge = CvBridge()
        self.pub_coord = self.create_publisher(Pose2D, '/yolo_detections', 10)
        
        self.timer = self.create_timer(0.5, self.test_loop)
        self.get_logger().info('--- CHE DO TEST: Bam q de thoat ---')

    def test_loop(self):
        if self.test_img is None:
            return

        frame_to_show   = self.test_img.copy()
        results         = self.model.predict(frame_to_show, verbose=False, conf=0.7)
        annotated_frame = results[0].plot()

        if results and results[0].boxes:
            num_objects = len(results[0].boxes)
            
            cv2.putText(annotated_frame, f"TOTAL: {num_objects}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            self.get_logger().info(f'--- Tim thay {num_objects} vat the ---')

            for i, box in enumerate(results[0].boxes):
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
                
                cx = int((x_min + x_max) / 2)
                cy = int((y_min + y_max) / 2)
                w  = x_max - x_min

                text_coord = f"X:{cx} Y:{cy}"
                cv2.putText(annotated_frame, text_coord, (x_min, y_max + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                self.get_logger().info(f' Object {i+1}: X={cx}, Y={cy}')
                
                msg       = Pose2D()
                msg.x     = float(cx)
                msg.y     = float(cy)
                msg.theta = float(w)
                self.pub_coord.publish(msg)
        else:
            self.get_logger().warn('Zero objects found.')

        cv2.imshow("KET QUA", annotated_frame)      
        key = cv2.waitKey(1) 
        
def main(args=None):
    rclpy.init(args=args)
    node = YOLOTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
