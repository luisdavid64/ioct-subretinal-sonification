import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np


import os
import glob

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class ImageSegmentationPublisher(Node):
    def __init__(self):
        super().__init__('image_segmentation_publisher')

        self.bridge = CvBridge()

        # ---- Parameters ----
        self.declare_parameter('image_dir', '')
        self.declare_parameter('seg_dir', '')

        self.image_dir = self.get_parameter('image_dir').get_parameter_value().string_value
        self.seg_dir = self.get_parameter('seg_dir').get_parameter_value().string_value

        if not self.image_dir:
            raise RuntimeError('image_dir parameter not set')

        # ---- Collect files ----
        self.image_files = sorted(
            glob.glob(os.path.join(self.image_dir, '*.png'))
        )

        print(len(self.image_files))

        if not self.image_files:
            raise RuntimeError(f'No images found in {self.image_dir}')

        self.seg_files = None
        if self.seg_dir:
            self.seg_files = sorted(
                glob.glob(os.path.join(self.seg_dir, '*.png'))
            )

            if len(self.seg_files) != len(self.image_files):
                self.get_logger().warn(
                    'Number of segmentations does not match images'
                )

        self.idx = 0

        # ---- Publishers ----
        self.image_pub = self.create_publisher(Image, '/gan/bscans', 10)
        self.seg_pub = self.create_publisher(Image, '/gan/segmentation', 10)

        self.timer = self.create_timer(1/10, self.publish)
        self.get_logger().info(
            f'Publishing {len(self.image_files)} images'
        )

    def publish(self):
        if self.idx >= len(self.image_files):
            self.get_logger().info('Finished publishing dataset')
            return

        # ---- Load image ----
        image_path = self.image_files[self.idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().error(f'Failed to read {image_path}')
            self.idx += 1
            return

        image_msg = self.bridge.cv2_to_imgmsg(
            image, encoding='bgr8'
        )

        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = 'camera'

        self.image_pub.publish(image_msg)

        # ---- Load segmentation if available ----
        if self.seg_files:
            seg_path = self.seg_files[self.idx]
            seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            # seg = seg * 60

            if seg is not None:
                seg_msg = self.bridge.cv2_to_imgmsg(
                    seg, encoding='mono8'
                )
                seg_msg.header = image_msg.header
                self.seg_pub.publish(seg_msg)

        self.get_logger().info(
            f'Published {os.path.basename(image_path)}'
        )

        self.idx += 1

def main():
    rclpy.init()
    node = ImageSegmentationPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()