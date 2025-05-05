#!/usr/bin/env python3

import os
import cv2
import math
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo

class LaneDetectorCV:
    def __init__(self):
        rospy.init_node('lane_detection_node', anonymous=True)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.camera_info_received = False

        rospy.Subscriber("zed2/zed_node/left/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("zed2/zed_node/left/image_rect_color", Image, self.img_callback, queue_size=1)

        # rospy.Subscriber("zed2/zed_node/right/camera_info", CameraInfo, self.camera_info_callback)
        # rospy.Subscriber("zed2/zed_node/right/image_rect_color", Image, self.img_callback, queue_size=1)

        self.pub_contrasted_image = rospy.Publisher("lane_detection/contrasted_image", Image, queue_size=1)
        self.pub_annotated = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_waypoints = rospy.Publisher("lane_detection/waypoints", Path, queue_size=1)
        self.pub_endgoal = rospy.Publisher("lane_detection/endgoal", PoseStamped, queue_size=1)

        self.estimated_lane_width_pixels = 700
        self.endgoal = None

        p1 = np.float32([[0.20*1280,0.75*720],[0,0.95*720],[0.80*1280,0.75*720],[1280,0.95*720]])
        p2 = np.float32([[0,0],[0,720],[1280,0],[1280,720]])
        self.M_bird = cv2.getPerspectiveTransform(p1, p2)
        self.M_bird_inv = cv2.getPerspectiveTransform(p2, p1)

    def camera_info_callback(self, msg):
        K = msg.K
        self.camera_matrix = np.array(K).reshape(3, 3)
        self.camera_info_received = True

    def img_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            #original
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 40, 255])
            lower_yellow = np.array([15, 127, 127])
            upper_yellow = np.array([35, 255, 255])

            #my version
            # lower_white = np.array([0, 0, 85])
            # upper_white = np.array([360, 5, 100])
            # lower_yellow = np.array([40, 50, 80])
            # upper_yellow = np.array([70, 100, 100])

            white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
            yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

            enhanced = cv2.bitwise_and(img, img, mask=combined_mask)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)

            bird = cv2.warpPerspective(binary, self.M_bird, (img.shape[1], img.shape[0]))
            bird_color = cv2.warpPerspective(enhanced, self.M_bird, (img.shape[1], img.shape[0]))

            path, left_boundary, right_boundary = self.generate_waypoints(bird)
            annotated = self.draw_waypoints(bird_color, path, left_boundary, right_boundary)
            self.publish_waypoints(path)

            self.pub_contrasted_image.publish(self.bridge.cv2_to_imgmsg(bird_color, "bgr8"))
            self.pub_annotated.publish(self.bridge.cv2_to_imgmsg(annotated, "bgr8"))

        except CvBridgeError as e:
            print(e)


    def compute_slope_angle(self, boundary):
        pts = [(x, y) for pt in boundary if pt is not None for x, y in [pt]]
        if len(pts) < 2:
            return None
        xs, ys = zip(*pts)
        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        if dx == 0:
            return None
        slope = dy / dx
        angle = math.atan(abs(slope))
        return angle
    

    def adjusted_width(self, angle):
        base_lane_width = self.estimated_lane_width_pixels
        if angle is None:
            return base_lane_width
        correction = 1 / max(abs(np.cos(np.pi/2 - angle)), 0.3)
        return base_lane_width * correction
    

    def generate_waypoints(self, lane_mask):
        """
        Generate navigation waypoints from lane mask with dynamic width correction.
        """
        path = Path()
        path.header.frame_id = "map"

        height, width = lane_mask.shape
        sampling_step = 5
        left_boundary = []
        right_boundary = []

        for y in range(height - 1, 0, -sampling_step):
            x_indices = np.where(lane_mask[y, :] > 0)[0]
            if len(x_indices) > 1:
                left_pts = x_indices[x_indices < width // 2]
                right_pts = x_indices[x_indices >= width // 2]

                left_pt = left_pts[0] if len(left_pts) > 0 else None
                right_pt = right_pts[-1] if len(right_pts) > 0 else None

                left_boundary.append((left_pt, y) if left_pt is not None else None)
                right_boundary.append((right_pt, y) if right_pt is not None else None)

            elif len(x_indices) == 1:
                x = x_indices[0]
                if x < width // 2:
                    left_boundary.append((x, y))
                    right_boundary.append(None)
                else:
                    left_boundary.append(None)
                    right_boundary.append((x, y))
            else:
                left_boundary.append(None)
                right_boundary.append(None)

        angle_left = self.compute_slope_angle(left_boundary)
        angle_right = self.compute_slope_angle(right_boundary)
        adjusted_width_left = self.adjusted_width(angle_left)
        adjusted_width_right = self.adjusted_width(angle_right)

        # ==== 建立 waypoints ====
        for lb, rb in zip(left_boundary, right_boundary):
            if lb and rb:
                x_center = (lb[0] + rb[0]) // 2
                y_center = lb[1]
            elif lb:
                x_center = int(lb[0] + adjusted_width_left // 2)
                y_center = lb[1]
            elif rb:
                x_center = int(rb[0] - adjusted_width_right // 2)
                y_center = rb[1]
            else:
                continue

            pt = PoseStamped()
            pt.pose.position.x = x_center
            pt.pose.position.y = y_center
            path.poses.append(pt)

        path.poses = path.poses[10:60]

        # ==== Endgoal ====
        if len(path.poses) > 0:
            xs = [p.pose.position.x for p in path.poses]
            ys = [p.pose.position.y for p in path.poses]
            median_x = np.median(xs)
            median_y = np.median(ys)

            self.endgoal = PoseStamped()
            self.endgoal.header = path.header
            self.endgoal.pose.position.x = median_x
            self.endgoal.pose.position.y = median_y
        else:
            self.endgoal = None

        return path, left_boundary, right_boundary


    def publish_waypoints(self, path):
        if self.camera_matrix is None:
            return

        for pose in path.poses:
            pose.pose.position.x, pose.pose.position.y, _ = self.image_to_world(
                pose.pose.position.x, pose.pose.position.y, self.camera_matrix)

        self.pub_waypoints.publish(path)

        if self.endgoal:
            self.endgoal.pose.position.x, self.endgoal.pose.position.y, _ = self.image_to_world(
                self.endgoal.pose.position.x, self.endgoal.pose.position.y, self.camera_matrix)
            # print(self.endgoal.pose.position.x, self.endgoal.pose.position.y)
            self.pub_endgoal.publish(self.endgoal)


    def draw_waypoints(self, img, path, left, right):
        img_draw = img.copy()
        for pose in path.poses:
            x, y = int(pose.pose.position.x), int(pose.pose.position.y)
            cv2.circle(img_draw, (x, y), 5, (0, 255, 255), -1)
        for lb in left:
            if lb:
                cv2.circle(img_draw, (int(lb[0]), int(lb[1])), 3, (255, 0, 0), -1)
        for rb in right:
            if rb:
                cv2.circle(img_draw, (int(rb[0]), int(rb[1])), 3, (0, 255, 0), -1)
        if self.endgoal:
            x, y = int(self.endgoal.pose.position.x), int(self.endgoal.pose.position.y)
            cv2.circle(img_draw, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(img_draw, "Endgoal", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return img_draw

    def image_to_world(self, u, v, camera_matrix, camera_height=1.85):
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        Z = camera_height
        bird_pt = np.array([[[u, v]]], dtype=np.float32)
        orig_pt = cv2.perspectiveTransform(bird_pt, self.M_bird_inv)[0][0]
        X = Z * (orig_pt[0] - cx) / fx
        Y = Z * (orig_pt[1] - cy) / fy
        return X, Y, Z

if __name__ == "__main__":
    try:
        detector = LaneDetectorCV()
        print("[INFO] Lane detector initialized.")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass