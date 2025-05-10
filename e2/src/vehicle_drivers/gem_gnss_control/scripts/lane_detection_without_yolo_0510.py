#!/usr/bin/env python3

import os
import cv2
import math
import numpy as np
import rospy
import warnings
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Header, Bool, Float32, Float64

class LaneDetectorCV:
    def __init__(self):
        rospy.init_node('lane_detection_node', anonymous=True)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.camera_info_received = False

        self.using_one_lane = 'left' # left, or right, or undesignated
        self.scale = 700 / 4

        rospy.Subscriber("zed2/zed_node/left/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("zed2/zed_node/left/image_rect_color", Image, self.img_callback, queue_size=1)

        # rospy.Subscriber("zed2/zed_node/right/camera_info", CameraInfo, self.camera_info_callback)
        # rospy.Subscriber("zed2/zed_node/right/image_rect_color", Image, self.img_callback, queue_size=1)

        self.prev_endgoal = None
        self.endgoal = None

        self.pub_contrasted_image = rospy.Publisher("lane_detection/contrasted_image", Image, queue_size=1)
        self.pub_annotated = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_waypoints = rospy.Publisher("lane_detection/waypoints", Path, queue_size=1)
        self.pub_endgoal = rospy.Publisher("lane_detection/endgoal", PoseStamped, queue_size=1)

        self.pub_waypoints_ic = rospy.Publisher("lane_detection/waypoints_image_coordinates", Path, queue_size=1)
        self.pub_endgoal_ic = rospy.Publisher("lane_detection/endgoal_image_coordinates", PoseStamped, queue_size=1)

        ###############################################################################
        # Obstacle detection
        ###############################################################################
        self.is_obs = False
        # rospy.Subscriber("/perception/obstacle_if", Bool, self.obs_detect_callback) # uncomment this line if you want to test obstacle avoidance
        self.turn_dir = 'right'  # or 'right', this can be adjusted by your own logic
        self.avoidance_curve_left = self.generate_curve('left')
        self.avoidance_curve_right = self.generate_curve('right')
        self.avoidance_step = 0
        self.last_avoidance_time = rospy.Time.now()
        self.pub_time = 0.5

        # self.estimated_lane_width_pixels = 700 # 500 and 750
        self.estimated_lane_width_pixels = 700

        p1 = np.float32([[0.15*1280,0.75*720],[0,0.95*720],[0.85*1280,0.75*720],[1280,0.95*720]])
        # p1 = np.float32([[0.20*1280,0.65*720],[0,0.95*720],[0.80*1280,0.65*720],[1280,0.95*720]])
        # p1 = np.float32([[0.25*1280,0.65*720],[0,0.95*720],[0.75*1280,0.65*720],[1280,0.95*720]])
        p2 = np.float32([[0,0],[0,720],[1280,0],[1280,720]])
        self.M_bird = cv2.getPerspectiveTransform(p1, p2)
        self.M_bird_inv = cv2.getPerspectiveTransform(p2, p1)

    def camera_info_callback(self, msg):
        K = msg.K
        self.camera_matrix = np.array(K).reshape(3, 3)
        self.camera_info_received = True

    def obs_detect_callback(self, msg):
        self.is_obs = msg

    def img_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # original
            lower_white = np.array([0, 0, 245]) # if you feel too noisy (capture too much white color), raise only the 3rd value; reduce only the 3rd value if lanes are too blury
            upper_white = np.array([180, 40, 255])
            # lower_yellow = np.array([15, 127, 127])
            # upper_yellow = np.array([35, 255, 255])

            # my version
            # lower_white = np.array([0, 0, 85])
            # upper_white = np.array([360, 5, 100])
            # lower_yellow = np.array([40, 50, 80])
            # upper_yellow = np.array([70, 100, 100])

            white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
            # yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

            # combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

            # enhanced = cv2.bitwise_and(img, img, mask=combined_mask)
            enhanced = cv2.bitwise_and(img, img, mask=white_mask)  # uncomment this line if you only want white
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)

            bird = cv2.warpPerspective(binary, self.M_bird, (img.shape[1], img.shape[0]))
            bird_color = cv2.warpPerspective(enhanced, self.M_bird, (img.shape[1], img.shape[0]))

            path, left_boundary, right_boundary = self.generate_waypoints(bird)
            annotated = self.draw_waypoints(bird_color.copy(), path, left_boundary, right_boundary)
            self.publish_waypoints(path)

            self.pub_contrasted_image.publish(self.bridge.cv2_to_imgmsg(bird_color, "bgr8"))
            self.pub_annotated.publish(self.bridge.cv2_to_imgmsg(annotated, "bgr8"))

        except CvBridgeError as e:
            print(e)

    def compute_slope_angles(self, boundary):
        valid_points = [(i, pt) for i, pt in enumerate(boundary) if pt is not None]
        if len(valid_points) < 3:
            return [None if pt is None else np.pi / 2 for pt in boundary]

        xs = np.array([pt[1][0] for pt in valid_points], dtype=np.float32)
        ys = np.array([pt[1][1] for pt in valid_points], dtype=np.float32)

        # Check data quality
        if np.any(np.isnan(xs)) or np.any(np.isnan(ys)) or np.all(xs == xs[0]):
            return [None if pt is None else np.pi / 2 for pt in boundary]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', np.RankWarning)  # Treat RankWarning as an exception
                coeffs = np.polyfit(xs, ys, 2)
                a, b, _ = coeffs
        except (np.RankWarning, np.linalg.LinAlgError):
            # fallback: compute slope from adjacent points
            angles = []
            n = len(boundary)
            for i in range(n):
                curr = boundary[i]
                if curr is None:
                    angles.append(None)
                    continue

                # Find next non-None point
                next_idx = i + 1
                while next_idx < n and boundary[next_idx] is None:
                    next_idx += 1

                if next_idx >= n:
                    angles.append(np.pi / 2)
                else:
                    x1, y1 = curr
                    x2, y2 = boundary[next_idx]
                    dx = x2 - x1
                    dy = y2 - y1
                    angle = np.pi / 2 if dx == 0 else abs(math.atan2(dy, dx))
                    angles.append(angle)
            return angles

        # If fitting succeeded, compute slope using derivative
        angles = []
        for pt in boundary:
            if pt is None:
                angles.append(None)
                continue
            x = pt[0]
            slope = 2 * a * x + b
            angle = abs(math.atan(slope))
            angles.append(angle)
        return angles

    def adjusted_width(self, angle):
        base_lane_width = self.estimated_lane_width_pixels
        if angle is None:
            return base_lane_width
        correction = 1 / max(np.sin(angle), 0.3)
        return base_lane_width * correction

    def generate_waypoints(self, lane_mask):
        """
        Generate navigation waypoints from lane mask or use avoidance curve if obstacle is detected.
        """
        if self.is_obs:
            # Choose avoidance curve based on direction
            curve = self.avoidance_curve_left if self.turn_dir == 'left' else self.avoidance_curve_right
            now = rospy.Time.now()
            if (now - self.last_avoidance_time).to_sec() >= self.pub_time:
                self.last_avoidance_time = now
                self.avoidance_step += 1

            path = Path()
            path.header.frame_id = "map"
            for i in range(self.avoidance_step, len(curve)):
                pt = PoseStamped()
                pt.header.frame_id = "map"
                pt.pose.position.x, pt.pose.position.y = int(curve[i][0]), int(curve[i][1])
                pt.pose.position.y -= 10 * (i - self.avoidance_step)
                path.poses.append(pt)

            if self.avoidance_step < len(curve):
                x, y = curve[self.avoidance_step]
            else:
                x, y = curve[-1]
                rospy.loginfo_throttle(1.0, "Final virtual waypoint has been published.")

            self.endgoal = PoseStamped()
            self.endgoal.header.frame_id = "map"
            self.endgoal.pose.position.x = int(x)
            self.endgoal.pose.position.y = int(y)

            rospy.loginfo_throttle(1.0, "Using avoidance curve.")
            return path, [], []

        # ================= Normal case: generate waypoints from lane_mask =================
        path = Path()
        path.header.frame_id = "map"
        height, width = lane_mask.shape
        sampling_step = 5
        left_boundary, right_boundary = [], []

        for y in range(height - 1, 0, -sampling_step):
            x_indices = np.where(lane_mask[y, :] > 0)[0]
            if len(x_indices) > 1:
                left_pts = x_indices[x_indices < width // 2]
                right_pts = x_indices[x_indices >= width // 2]
                left_boundary.append((left_pts[0], y) if len(left_pts) > 0 else None)
                right_boundary.append((right_pts[-1], y) if len(right_pts) > 0 else None)
            elif len(x_indices) == 1:
                x = x_indices[0]
                left_boundary.append((x, y) if x < width // 2 else None)
                right_boundary.append((x, y) if x >= width // 2 else None)
            else:
                left_boundary.append(None)
                right_boundary.append(None)

        angles_left = self.compute_slope_angles(left_boundary)
        angles_right = self.compute_slope_angles(right_boundary)

        weight = np.clip(len(left_boundary) / (len(left_boundary) + len(right_boundary)), 0.3, 0.7)
        for lb, rb, al, ar in zip(left_boundary, right_boundary, angles_left, angles_right):
            if self.using_one_lane == 'left':
                if lb:
                    x_center = int(lb[0] + self.adjusted_width(al) // 2)
                    y_center = lb[1]
                else:
                    continue
            elif self.using_one_lane == 'right':
                if rb:
                    x_center = int(rb[0] - self.adjusted_width(ar) // 2)
                    y_center = rb[1]
                else:
                    continue
            elif self.using_one_lane == 'undesignated':
                if lb and rb:
                    x_center = int(lb[0] * weight + rb[0] * (1.0 - weight))
                    y_center = max(lb[1], rb[1])
                elif lb:
                    x_center = int(lb[0] + self.adjusted_width(al) // 2)
                    y_center = lb[1]
                elif rb:
                    x_center = int(rb[0] - self.adjusted_width(ar) // 2)
                    y_center = rb[1]
                else:
                    continue

            pt = PoseStamped()
            pt.pose.position.x = x_center
            pt.pose.position.y = y_center
            path.poses.append(pt)

        path.poses = path.poses[5:30]

        # Endgoal computation
        if len(path.poses) > 0:
            xs = [p.pose.position.x for p in path.poses]
            ys = [p.pose.position.y for p in path.poses]
            self.endgoal = PoseStamped()
            self.endgoal.header = path.header
            self.endgoal.pose.position.x = np.median(xs)
            self.endgoal.pose.position.y = np.median(ys) 
        else:
            self.endgoal = None

        return path, left_boundary, right_boundary
    

    def generate_curve(self, direction='left', num_points=30):
        """
        Generate an offset curve for obstacle avoidance.
        The start point is (0, 0), end point is (+/-2, -4),
        x uses sine curve, y uses linear descent.
        """
        if direction == 'left':
            sign = -1
        elif direction == 'right':
            sign = 1
        else:
            raise ValueError("direction must be 'left' or 'right'")

        # image coordinate
        if self.camera_matrix:
            cx = self.camera_matrix[0, 2]
        else:
            cx = 1225 / 2
        x_curve = cx + sign * (1.0 * self.scale * np.sin(np.linspace(0, 2 * np.pi, num_points)))
        y_curve = np.linspace(720 - 1.2 * self.scale, 720 - 1.2 * self.scale, num_points)

        return list(zip(x_curve, y_curve))

    def publish_waypoints(self, waypoints):
        """
        Publish waypoints and endgoal, assuming already generated (with or without obstacle).
        """
        self.avoidance_step = 0 if not self.is_obs else self.avoidance_step

        # Publish waypoints in image coordinates
        self.pub_waypoints_ic.publish(waypoints)

        # Publish waypoints in world coordinates
        for pose in waypoints.poses:
            x, y, _ = self.image_to_world(pose.pose.position.x, pose.pose.position.y, self.camera_matrix)
            pose.pose.position.x = np.clip(x, -1.0, 1.0)
            pose.pose.position.y = y
        self.pub_waypoints.publish(waypoints)

        if self.endgoal is not None:
            # Publish endgoal in image coordinates
            self.pub_endgoal_ic.publish(self.endgoal)

            # Publish endgoal in world coordinates
            x, y, _ = self.image_to_world(self.endgoal.pose.position.x, self.endgoal.pose.position.y, self.camera_matrix)
            self.endgoal.pose.position.x = x
            self.endgoal.pose.position.y = y
            self.pub_endgoal.publish(self.endgoal)

    def draw_waypoints(self, img, waypoints, left_boundary, right_boundary):
        """
        Draw waypoints and lane boundaries on image for visualization.
        """
        # Draw waypoints as yellow circles
        for pose in waypoints.poses:
            x, y = int(pose.pose.position.x), int(pose.pose.position.y)
            cv2.circle(img, (x, y), radius=5, color=(0, 180, 180), thickness=-1)

        # Draw left boundary as blue circles
        for lb in left_boundary:
            if lb is not None:
                x, y = int(lb[0]), int(lb[1])
                cv2.circle(img, (x, y), radius=3, color=(255, 0, 0), thickness=-1)

        # Draw right boundary as green circles
        for rb in right_boundary:
            if rb is not None:
                x, y = int(rb[0]), int(rb[1])
                cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        # Draw end goal as red circle with label
        if self.endgoal is not None:
            ex, ey = int(self.endgoal.pose.position.x), int(self.endgoal.pose.position.y)
            cv2.circle(img, (ex, ey), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, "Endgoal", (ex + 15, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return img

    def image_to_world(self, u, v, camera_matrix, camera_height=1.85):
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Use perspective transform to map bird's-eye view coordinates back to original image coordinates
        bird_pt = np.array([[[u, v]]], dtype=np.float32)
        orig_pt = cv2.perspectiveTransform(bird_pt, self.M_bird_inv)[0][0]
        x_img, y_img = orig_pt

        Z = camera_height  # Relative height of the ground to the camera
        X = (x_img - cx) * Z / fx
        Y = (cy - y_img) * Z / fy - (cy - 720) * Z / fy + 1.7 + 1.76  # Note: upward is Y+

        return X, Y, Z


if __name__ == "__main__":
    try:
        detector = LaneDetectorCV()
        print("[INFO] Lane detector initialized. Now using without YOLO.")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass