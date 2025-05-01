#!/usr/bin/env python3

#================================================================
# File name: lane_detection.py                                                                  
# Description: learning-based lane detection module                                                            
# Author: Siddharth Anand
# Email: sanand12@illinois.edu                                                                 
# Date created: 08/02/2021                                                                
# Date last modified: 03/14/2025
# Version: 1.0                                                                   
# Usage: python lane_detection.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import cv2
import csv
import math
import time
import torch
import threading
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt

from filters import OnlineFilter

# ROS Headers
import rospy
from nav_msgs.msg import Path

# GEM Sensor Headers
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Header, Bool, Float32, Float64
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva

# GEM PACMod Headers
from geometry_msgs.msg import PoseStamped
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

###############################################################################
# Lane Detection Node
# 
# This module implements deep learning-based lane detection using YOLOPv2.
# It processes images from a camera, identifies lane markings, and publishes
# waypoints for autonomous navigation.
###############################################################################

class LaneNetDetector:
    """
    Main class for lane detection using YOLOPv2 neural network.
    
    This class handles:
    1. Image preprocessing and enhancement
    2. Deep learning model inference
    3. Lane detection and boundary identification
    4. Waypoint generation for vehicle navigation
    5. Visual feedback through annotated images
    """
    
    def __init__(self, path_to_weights='../../../weights/yolopv2.pt'):
        """
        Initialize the lane detection node with model, parameters and ROS connections.
        
        Sets up:
        - Frame buffering for stable detection
        - Deep learning model (YOLOPv2)
        - ROS publishers and subscribers
        - Image processing parameters
        """
        if not os.path.exists(path_to_weights):
            raise FileNotFoundError(f"Model weights not found at {path_to_weights}")

        # Frame buffer for batch processing to increase efficiency
        self.frame_buffer = []
        self.buffer_size = 4  # Process 4 frames at once for better throughput
        
        # Initialize ROS node
        rospy.init_node('lane_detection_node', anonymous=True)
        
        # Image processing utilities and state variables
        self.bridge = CvBridge()  # Converts between ROS Image messages and OpenCV images
        self.prev_left_boundary = None  # Store previous lane boundary for smoothing
        self.estimated_lane_width_pixels = 1000  # Approximate lane width in image pixels
        self.prev_waypoints = None  # Previous waypoints for temporal consistency
        self.endgoal = None  # Target point for navigation
        
        ###############################################################################
        # Deep Learning Model Setup
        ###############################################################################
        
        # Set up compute device (GPU if available, otherwise CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load pre-trained YOLOPv2 model for lane detection
        self.model = torch.jit.load(path_to_weights)
        self.half = self.device != 'cpu'  # Use half precision for faster inference on GPU
        
        # Configure model for inference
        if self.half:
            self.model.half()  # Convert model to half precision
            
        self.model.to(self.device).eval()  # Move model to device and set to evaluation mode
        
        # Commented out stop sign detection model code
        # self.stop_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, trust_repo=True)
        # self.stop_model.to(self.device).eval()
        
        ###############################################################################
        # Camera and Control Parameters
        ###############################################################################
        
        self.camera_matrix = None  # will be initialized in camera_info callback
        self.camera_info_received = False
        rospy.Subscriber("zed2/zed_node/left/camera_info", CameraInfo, self.camera_info_callback)
        # zed2/zed_node/rgb/camera_info
        # zed2/zed_node/rgb_raw/camera_info

        p1 = np.float32([[0.35*1280,0.6*720],[0.15*1280,720],[0.65*1280,0.6*720],[0.85*1280,720]])
        p2 = np.float32([[0,0],[0,720],[1280,0],[1280,720]])
        self.M_bird = cv2.getPerspectiveTransform(p1, p2)
        self.M_bird_inv = cv2.getPerspectiveTransform(p2, p1)

        self.Focal_Length = 528  # origin: 800; Camera focal length in pixels; HD 720; reference: https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view
        self.Real_Height_SS = .75  # Height of stop sign in meters (not used currently)
        self.Brake_Distance = 5  # Distance at which to apply brakes (not used currently)
        self.Brake_Duration = 3  # Duration to hold brakes (not used currently)
        
        ###############################################################################
        # ROS Communication Setup
        ###############################################################################
        
        # Subscribe to camera feed
        self.sub_image = rospy.Subscriber('zed2/zed_node/left/image_rect_color', Image, self.img_callback, queue_size=1)
        # note that oak/rgb/image_raw is the topic name for the GEM E4. If you run this on the E2, you will need to change the topic name
        # possible ros topic for E2:
        # zed2/zed_node/rgb/image_rect_color
        # zed2/zed_node/rgb_raw/image_rect_color
        
        # Publishers for visualization and control
        self.pub_contrasted_image = rospy.Publisher("lane_detection/contrasted_image", Image, queue_size=1)
        self.pub_annotated = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_waypoints = rospy.Publisher("lane_detection/waypoints", Path, queue_size=1)
        self.pub_endgoal = rospy.Publisher("lane_detection/endgoal", PoseStamped, queue_size=1)

        ###############################################################################
        # Vehicle state Setup
        ###############################################################################
        # self.gnss_sub   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        # self.lat        = 0.0
        # self.lon        = 0.0
        
        # self.olat       = 40.0928563
        # self.olon       = -88.2359994

        # self.local_x_curr = 0.0
        # self.local_y_curr = 0.0
        ###############################################################################
        # Visualization Setup
        ###############################################################################
        # self.endgoal_history = []   # 用來儲存時間戳與endgoal位置
        # self.visualization_thread = threading.Thread(target=self.update_plot, daemon=True)
        # self.visualization_thread.start()
        # self.plot_running = True

        # self.output_dir = "./lane_detection_output"
        # os.makedirs(self.output_dir, exist_ok=True)


    def img_callback(self, img):
        """
        Process incoming camera images to detect lanes and generate waypoints.
        
        This function:
        1. Converts ROS image to OpenCV format
        2. Enhances image using color filtering and contrast
        3. Preprocesses image for neural network
        4. Adds image to buffer for batch processing
        5. Performs inference when buffer is full
        6. Generates and publishes waypoints and visualizations
        
        Args:
            img: ROS Image message from camera
        """
        try:
            # Convert ROS Image to OpenCV format
            img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            
            ###############################################################################
            # Image Enhancement Pipeline
            ###############################################################################
            
            # Convert to HSV color space for better color segmentation
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Define white color range for lane markings
            lower_white = np.array([210, 210, 210])
            upper_white = np.array([255, 255, 255])
            
            # Create masks to enhance white lane markings
            white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
            non_white_mask = cv2.bitwise_not(white_mask)
            
            # Remove non-white areas and convert to grayscale
            img_no_white = cv2.bitwise_and(img, img, mask=non_white_mask)
            img_gray = cv2.cvtColor(img_no_white, cv2.COLOR_BGR2GRAY)
            
            # Apply contrast enhancement using thresholding
            threshold = 180
            mask = img_gray >= threshold
            dimmed_gray = (img_gray * 0).astype(np.uint8)  # Reduce brightness of non-lane areas
            dimmed_gray[mask] = img_gray[mask]  # Keep bright areas at original intensity
            
            # Convert back to BGR for visualization
            contrasted_img = cv2.cvtColor(dimmed_gray, cv2.COLOR_GRAY2BGR)

            # Apply region of interest
            contrasted_img = self.region_of_interest(contrasted_img, mode='bird')
            
            # Publish enhanced image for debugging
            contrasted_image_msg = self.bridge.cv2_to_imgmsg(contrasted_img, "bgr8")
            self.pub_contrasted_image.publish(contrasted_image_msg)
            
            ###############################################################################
            # Model Inference Pipeline
            ###############################################################################
            
            # Preprocess image for neural network
            img_tensor = self.preprocess_frame(contrasted_img)
            
            # Add to buffer for batch processing
            self.frame_buffer.append((contrasted_img, img_tensor))
            
            # When buffer is full, process batch for efficiency
            if len(self.frame_buffer) >= self.buffer_size:
                # Separate original images and tensors
                original_images, tensors = zip(*self.frame_buffer)
                
                # Stack tensors into a batch
                batch = torch.stack(tensors).to(self.device)
                
                # Clear buffer after processing
                self.frame_buffer.clear()
                
                # Run inference on batch
                with torch.no_grad():
                    [pred, anchor_grid], seg, ll = self.model(batch)
                
                # Process each result in the batch
                for i, contrasted_img in enumerate(original_images):
                    # Generate waypoints and annotate image
                    annotated_img = self.detect_lanes(seg[i], ll[i], contrasted_img)
                    
                    # Publish annotated image
                    annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
                    self.pub_annotated.publish(annotated_image_msg)
                    
        except CvBridgeError as e:
            print(e)

    def preprocess_frame(self, img):
        """
        Preprocess image for neural network input.
        
        Steps:
        1. Resize image with letterboxing to maintain aspect ratio
        2. Convert BGR to RGB and change channel order
        3. Convert to tensor and normalize
        
        Args:
            img: OpenCV image (BGR format)
            
        Returns:
            PyTorch tensor ready for model inference
        """
        # Resize with letterboxing to model input size (384x640)
        img_resized, _, _ = self.letterbox(img, new_shape=(384, 640))
        
        # Convert BGR to RGB and change to channel-first format
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
        
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_resized))
        
        # Convert to half precision if using GPU
        img_tensor = img_tensor.half() if self.half else img_tensor.float()
        
        # Normalize pixel values to 0-1
        img_tensor /= 255.0
        
        return img_tensor

    def camera_info_callback(self, msg):
        """
        get camera info and save it as a 3x3 matrix for image_to_world
        """
        K = msg.K  # camera intrinsic matrix is a one-dimentional list with a length of 9
        self.camera_matrix = np.array(K).reshape(3, 3)
        self.camera_info_received = True

    # def inspva_callback(self, inspva_msg):
    #     self.lat     = inspva_msg.latitude  # latitude
    #     self.lon     = inspva_msg.longitude # longitude

    # def wps_to_local_xy(self, lon_wp, lat_wp):
    #     # convert GNSS waypoints into local fixed frame reprented in x and y
    #     lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
    #     return lon_wp_x, lat_wp_y 

    def image_to_world(self, u, v, camera_matrix, camera_height=1.85): # 1.85 meter
        """
        Convert image coordinates to world coordinates using pinhole camera model.
        
        This function projects a point from the image plane to 3D world coordinates
        assuming a flat ground plane.
        
        Args:
            u, v: Image coordinates (pixels)
            camera_matrix: Camera intrinsic parameters
            camera_height: Height of camera above ground plane
            
        Returns:
            X, Y, Z: World coordinates (meters)
        """
        # Extract camera intrinsic parameters
        fx = camera_matrix[0, 0]  # Focal length in x direction
        fy = camera_matrix[1, 1]  # Focal length in y direction
        cx = camera_matrix[0, 2]  # Principal point x coordinate
        cy = camera_matrix[1, 2]  # Principal point y coordinate
        
        # Assume ground plane is at z = -camera_height
        Z = -camera_height
        
        bird_eye_pts = np.array([[[u, v]]], dtype=np.float32)  # shape: (1, 1, 2)

        # from bird-view image coordinate transfer back to original image coordinate
        original_pts = cv2.perspectiveTransform(bird_eye_pts, self.M_bird_inv)
        X, Y = original_pts[0, 0]

        # Project using pinhole camera model
        X = Z * (X - cx) / fx
        Y = Z * (Y - cy) / fy
        print("x: ", X, "   y: ", Y)
        return X, Y, Z

    def detect_lanes(self, seg, ll, img):
        """
        Process neural network output to detect lanes and generate waypoints.
        
        Steps:
        1. Extract drivable area mask from segmentation output
        2. Extract lane line mask from lane detection output
        3. Generate waypoints based on detected lanes
        4. Draw waypoints on image for visualization
        
        Args:
            seg: Segmentation output from neural network
            ll: Lane line output from neural network
            img: Original image for annotation
            
        Returns:
            Annotated image with waypoints and lane boundaries
        """
        # Extract drivable area mask from segmentation output
        da_seg_mask = driving_area_mask(seg)
        
        # Extract lane line mask with confidence threshold
        ll_seg_mask = lane_line_mask(ll, threshold=0.2)
        
        # Generate waypoints from lane line mask
        waypoints, left_boundary, right_boundary = self.generate_waypoints(ll_seg_mask)
        
        # Draw waypoints on image for visualization
        img_with_waypoints = self.draw_waypoints(img.copy(), waypoints, left_boundary, right_boundary)
        
        # Publish waypoints for vehicle control
        self.publish_waypoints(waypoints)
        
        return img_with_waypoints

    def region_of_interest(self, img, mode='bird'):
        """
        Apply a region of interest mask to focus on relevant portion of image.
        
        This function creates a polygon mask to focus processing on the left
        half of the image where the lane is expected to be.
        
        Args:
            img: Input image
            
        Returns:
            Masked image with only the region of interest visible
        """

        # Get image dimensions
        height, width = img.shape[:2]

        if mode == 'bird':
            output = cv2.warpPerspective(img, self.M_bird, (width, height))

            return output
        
        elif mode == 'bird_inv':
            output = cv2.warpPerspective(img, self.M_bird_inv, (width, height))

            return output

        elif mode == 'left':
            # Create empty mask
            mask = np.zeros_like(img)
            
            # Define polygon for left half of image
            polygon = np.array([[(height, 0), (0, 0), (0, width // 2), (height, width // 2)]], np.int32)
            
            # Fill polygon with white
            cv2.fillPoly(mask, polygon, 255)
            
            # Apply mask to image
            masked_image = cv2.bitwise_and(img, mask)

            return masked_image
        
        elif mode == 'right':
            # Create empty mask
            mask = np.zeros_like(img)
            
            # Define polygon for left half of image
            polygon = np.array([[(height, width // 2), (0, width // 2), (0, width), (height, width)]], np.int32)
            
            # Fill polygon with white
            cv2.fillPoly(mask, polygon, 255)
            
            # Apply mask to image
            masked_image = cv2.bitwise_and(img, mask)

            return masked_image
        
        else:
            raise Exception("undefined mode for region_of_interest.")


    def generate_waypoints(self, lane_mask):
        """
        Generate navigation waypoints from lane mask.
        
        This function:
        1. Scans the lane mask from bottom to top
        2. Identifies the left lane boundary
        3. Estimates center of lane using offset from boundary
        4. Creates waypoints for vehicle to follow
        
        Args:
            lane_mask: Binary mask of detected lane markings
            
        Returns:
            path: ROS Path message with waypoints
            left_boundary: List of points representing left lane boundary
        """
        # Create ROS Path message
        path = Path()
        path.header.frame_id = "map"
        
        # Get dimensions of lane mask
        height, width = lane_mask.shape
        
        # Sampling parameters
        sampling_step = 10  # Sample every 10 pixels vertically
        left_boundary = []  # Store left boundary points
        right_boundary = []  # Store left boundary points
        offset_pixels = 300  # Estimated distance from left boundary to lane center
        
        ###############################################################################
        # Scan Lane Mask and Extract Boundaries
        ###############################################################################
        
        # # Scan from bottom to top of image
        # for y in range(height - 1, 0, -sampling_step):
        #     # Find all x coordinates where lane marking exists at this y
        #     x_indices = np.where(lane_mask[y, :] > 0)[0]
            
        #     if len(x_indices) > 0:
        #         # Use leftmost point as left boundary
        #         x_left = x_indices[0]
        #         # Store with slight offset for better tracking
        #         left_boundary.append((x_left - 40, y))
        #     else:
        #         # No lane marking found at this y
        #         left_boundary.append(None)
                
        # # Filter boundaries to get continuous line segments
        # left_boundary = self.filter_continuous_boundary(left_boundary)

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

        ###############################################################################
        # Generate Waypoints from Boundaries
        ###############################################################################
        
        # # Create waypoints from left boundary with offset
        # for lb in left_boundary:
        #     if lb:  # If boundary point exists
        #         # Calculate lane center by adding offset to left boundary
        #         x_center = lb[0] + offset_pixels
        #         y = lb[1]
                
        #         # Create ROS PoseStamped message for waypoint
        #         point = PoseStamped()
        #         point.pose.position.x = x_center
        #         point.pose.position.y = y
        #         path.poses.append(point)
                
        # # Use only the first 7 waypoints (closest to vehicle)
        # path.poses = path.poses[:7]

        for lb, rb in zip(left_boundary, right_boundary):
            if lb and rb:
                x_center = (lb[0] + rb[0]) // 2
                y_center = lb[1]  # y values should be the same
            elif lb:
                x_center = lb[0] + self.estimated_lane_width_pixels // 2
                y_center = lb[1]
            elif rb:
                x_center = rb[0] - self.estimated_lane_width_pixels // 2
                y_center = rb[1]
            else:
                continue

            pt = PoseStamped()
            pt.pose.position.x = x_center
            pt.pose.position.y = y_center
            path.poses.append(pt)

        # # Use only the first 15 waypoints (closest to vehicle)
        path.poses = path.poses[:15]
        
        ###############################################################################
        # Calculate End Goal (Target Point)
        ###############################################################################
        
        if len(path.poses) > 0:
            # Extract x and y coordinates from all waypoints
            xs = [p.pose.position.x for p in path.poses]
            ys = [p.pose.position.y for p in path.poses]
            
            # Use median for stability
            median_x = np.median(xs)
            median_y = np.median(ys)
            
            # Create end goal pose
            self.endgoal = PoseStamped()
            self.endgoal.header = path.header
            self.endgoal.pose.position.x = median_x
            self.endgoal.pose.position.y = median_y
        else:
            # No waypoints found
            self.endgoal = None
            
        return path, left_boundary, right_boundary

    def filter_continuous_boundary(self, boundary):
        """
        Filter boundary points to get continuous line segments.
        
        This helps remove noise and discontinuities in the detected lane boundary.
        Points with large horizontal gaps are considered discontinuities.
        
        Args:
            boundary: List of boundary points (x,y) or None
            
        Returns:
            List of filtered boundary points
        """
        max_gap = 60  # Maximum allowed horizontal gap between consecutive points
        continuous_boundary = []
        previous_point = None
        
        for point in boundary:
            if point is not None:
                if previous_point is None or abs(point[0] - previous_point[0]) <= max_gap:
                    # Point is continuous with previous point
                    continuous_boundary.append(point)
                    previous_point = point
                else:
                    # Gap too large, start new segment
                    continuous_boundary.append(None)
                    previous_point = None
            else:
                # No point detected
                continuous_boundary.append(None)
                previous_point = None
                
        return continuous_boundary

    def publish_waypoints(self, waypoints):
        """
        Publish waypoints and end goal for vehicle control.
        
        Args:
            waypoints: ROS Path message with waypoints
        """
        # Publish waypoints
        # Convert the waypoints into world coordinate first
        for pose in waypoints.poses:
            pose.pose.position.x, pose.pose.position.y, _ = self.image_to_world(pose.pose.position.x, pose.pose.position.y, self.camera_matrix)
        self.pub_waypoints.publish(waypoints)
        
        # Publish end goal if available
        if self.endgoal is not None:
            self.endgoal.pose.position.x, self.endgoal.pose.position.y, _ = self.image_to_world(self.endgoal.pose.position.x, self.endgoal.pose.position.y, self.camera_matrix)
            self.pub_endgoal.publish(self.endgoal)
        
        # self.local_x_curr, self.local_y_curr = self.wps_to_local_xy(self.lon, self.lat)
        # # record time and endgoal
        # self.endgoal_history.append((
        #     rospy.get_time(),  # 取得目前的ROS時間（浮點秒數）
        #     self.local_x_curr + self.endgoal.pose.position.x,
        #     self.local_y_curr + self.endgoal.pose.position.y
        # ))


    def draw_waypoints(self, img, waypoints, left_boundary, right_boundary):
        """
        Draw waypoints and lane boundaries on image for visualization.
        
        Args:
            img: Image to draw on
            waypoints: Path message containing waypoints
            left_boundary: List of points representing left lane boundary
            
        Returns:
            Annotated image
        """
        # Draw waypoints as yellow circles
        for pose in waypoints.poses:
            point = np.array([[[pose.pose.position.x, pose.pose.position.y]]], dtype=np.float32)
            point_img = cv2.perspectiveTransform(point, self.M_bird_inv)            
            # x, y = int(point_img[0, 0, 0]), int(point_img[0, 0, 1])
            x, y = int(pose.pose.position.x), int(pose.pose.position.y)
            cv2.circle(img, (x, y), radius=5, color=(0, 255, 255), thickness=-1)
        
        # Draw left boundary as blue circles
        for lb in left_boundary:
            if lb is not None:
                point = np.array([[[lb[0], lb[1]]]], dtype=np.float32)
                point_img = cv2.perspectiveTransform(point, self.M_bird_inv)
                # x, y = int(point_img[0, 0, 0]), int(point_img[0, 0, 1])
                x, y = int(lb[0]), int(lb[1])
                cv2.circle(img, (x, y), radius=3, color=(255, 0, 0), thickness=-1)

        # Draw right boundary as green circles
        for rb in right_boundary:
            if rb is not None:
                point = np.array([[[rb[0], rb[1]]]], dtype=np.float32)
                point_img = cv2.perspectiveTransform(point, self.M_bird_inv)
                # x, y = int(point_img[0, 0, 0]), int(point_img[0, 0, 1])
                x, y = int(rb[0]), int(rb[1])
                cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
        
        # Draw end goal as red circle with label
        if self.endgoal is not None:
            point = np.array([[[self.endgoal.pose.position.x, self.endgoal.pose.position.y]]], dtype=np.float32)
            point_img = cv2.perspectiveTransform(point, self.M_bird_inv)            
            ex, ey = int(point_img[0, 0, 0]), int(point_img[0, 0, 1])
            ex, ey = int(self.endgoal.pose.position.x), int(self.endgoal.pose.position.y)
            cv2.circle(img, (ex, ey), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, "Endgoal", (ex + 15, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        return img

    def letterbox(self, img, new_shape=(384, 640), color=(114, 114, 114)):
        """
        Resize image with letterboxing to maintain aspect ratio.
        
        This is important for neural network input to prevent distortion.
        
        Args:
            img: Input image
            new_shape: Target shape (height, width)
            color: Padding color
            
        Returns:
            resized_img: Resized and padded image
            ratio: Scale ratio (used for inverse mapping)
            padding: Padding values (dw, dh)
        """
        # Original shape
        shape = img.shape[:2]
        
        # Handle single dimension input
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Calculate scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        
        # Calculate new unpadded dimensions
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        # Calculate padding
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        # Resize image
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, ratio, (dw, dh)
    

    # def update_plot(self):
    #     plt.ion()  # 即時互動模式
    #     fig, ax = plt.subplots()
    #     line, = ax.plot([], [], marker='o', label='Endgoal Trajectory')

    #     ax.set_xlabel('Position X (meters)')
    #     ax.set_ylabel('Position Y (meters)')
    #     ax.set_title('Endgoal Trajectory (Real-Time)')
    #     ax.grid(True)
    #     ax.legend()

    #     while self.plot_running and not rospy.is_shutdown():
    #         if len(self.endgoal_history) > 0:
    #             _, xs, ys = zip(*self.endgoal_history)
    #             xs = np.array(xs)
    #             ys = np.array(ys)

    #             line.set_data(xs, ys)
    #             ax.relim()
    #             ax.autoscale_view()
    #             fig.canvas.draw()
    #             fig.canvas.flush_events()

    #         rospy.sleep(0.5)  # 每0.5秒更新一次

    # def save_endgoal_plot(self):
    #     if len(self.endgoal_history) == 0:
    #         print("No endgoal history to save.")
    #         return

    #     # 整理紀錄資料
    #     times, xs, ys = zip(*self.endgoal_history)
    #     times = np.array(times)
    #     xs = np.array(xs)
    #     ys = np.array(ys)

    #     start_time = times[0]
    #     times = times - start_time  # 讓時間從0秒開始

    #     # 畫出 Position X vs Position Y
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(xs, ys, marker='o', label='Endgoal trajectory')
    #     plt.xlabel('Position X (meters)')
    #     plt.ylabel('Position Y (meters)')
    #     plt.title('Endgoal Trajectory Over Time')
    #     plt.grid(True)
    #     plt.legend()

    #     # 每隔5秒標記一次時間
    #     for i in range(len(times)):
    #         if times[i] % 5 < 0.5:  # 允許小誤差，比如 5.1秒也可以標
    #             plt.text(xs[i], ys[i], f"{int(times[i])}s", fontsize=8, color='red')

    #     # 儲存
    #     save_path = os.path.join(self.output_dir, "final_endgoal_plot.png")
    #     plt.savefig(save_path)
    #     plt.close()
    #     print(f"Saved final static plot to {save_path}")

###############################################################################
# Neural Network Output Processing Functions
###############################################################################

def driving_area_mask(seg):
    """
    Extract drivable area mask from segmentation output.
    
    Args:
        seg: Segmentation output tensor from neural network
        
    Returns:
        Binary mask of drivable area
    """
    # Handle different tensor shapes
    if len(seg.shape) == 4:
        # Batch of images
        da_predict = seg[:, :, 12:372, :]
    elif len(seg.shape) == 3:
        # Single image
        seg = seg.unsqueeze(0)  # Add batch dimension
        da_predict = seg[:, :, 12:372, :]
    else:
        raise ValueError(f"Unexpected tensor shape in driving_area_mask: {seg.shape}")
    
    # Upscale mask to original resolution
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=2, mode='bilinear', align_corners=False)
    
    # Convert to binary mask (argmax across channels)
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    
    # Convert to numpy array
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
    
    return da_seg_mask

def lane_line_mask(ll, threshold):
    """
    Extract lane line mask from lane detection output.
    
    Args:
        ll: Lane line detection tensor from neural network
        threshold: Confidence threshold for lane detection
        
    Returns:
        Binary mask of lane lines
    """
    # Handle different tensor shapes
    if len(ll.shape) == 4:
        # Batch of images
        ll_predict = ll[:, :, 12:372, :]
    elif len(ll.shape) == 3:
        # Single image
        ll = ll.unsqueeze(0)  # Add batch dimension
        ll_predict = ll[:, :, 12:372, :]
    else:
        raise ValueError(f"Unexpected tensor shape in lane_line_mask: {ll.shape}")
    
    # Upscale mask to original resolution
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=2, mode='bilinear', align_corners=False)
    
    # Apply threshold to get binary mask
    ll_seg_mask = (ll_seg_mask > threshold).int().squeeze(1)
    
    # Convert to numpy array
    ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy().astype(np.uint8)
    
    # Dilate mask to fill gaps and strengthen lane line detection
    kernel = np.ones((2, 2), np.uint8)
    ll_seg_mask = cv2.dilate(ll_seg_mask, kernel, iterations=1)
    
    return ll_seg_mask


###############################################################################
# Main Entry Point
###############################################################################

if __name__ == "__main__":
    try:
        # Create detector instance
        detector = LaneNetDetector()
        
        # Keep node running until shutdown
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    # finally:
        # detector.plot_running = False  # 通知plot執行緒結束
        # rospy.sleep(1.0)  # 確保畫圖執行緒完全停止
        # detector.save_endgoal_plot()   # 儲存最後的靜態圖