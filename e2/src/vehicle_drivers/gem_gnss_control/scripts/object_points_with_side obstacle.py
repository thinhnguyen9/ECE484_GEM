#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32MultiArray
import numpy as np
# from scipy.spatial import distance



class ObstacleDetector:
    def __init__(self):
        rospy.init_node('obstacle_detector')

        # Target frame
        self.target_frame = "os_sensor"

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.msg_to_pub = Float32MultiArray()
        self.if_obs=Int32MultiArray()
        self.if_obs.data=[0,0] # [False,Flase]
        self.obs_direction = 'left' # negative , positive : right
        # Subscribe to point cloud
        rospy.Subscriber("/ouster/points", PointCloud2, self.cloud_cb)

    def cloud_cb(self, cloud_msg):
        try:
            # Transform point cloud to target frame
            trans_cloud = self.tf_buffer.transform(
                cloud_msg,
                self.target_frame,
                timeout=rospy.Duration(0.5)
            )

            # Parse transformed cloud
            points = pc2.read_points(trans_cloud, field_names=("x", "y", "z"), skip_nans=True)

            # Filter points within a box in front of the vehicle
            front_obstacles = []
            side_obstacles = []
            for x, y, z in points:
                if 3 < x < 5 and abs(y) < 1.0 and -1.3 < z < 1.5:
                    front_obstacles.append((x, y, z))
                if self.obs_direction == 'left':
                    if -2 < x < 2 and -4.5 < y < -2 and -1.3 < z < 1:
                        side_obstacles.append((x,y,z))
                elif self.obs_direction == 'right':
                    if -2 < x < 2 and 2 < y < 4.5 and -1.3 < z < 1:
                        side_obstacles.append((x,y,z))
            if front_obstacles:
                boxes = self.get_bounding_boxes(front_obstacles)
                if boxes:
                    self.if_obs.data[0]=1
                flat_data = []
                for box in boxes:
                    for pt in box:
                        flat_data.extend(pt)  # x, y, z

                self.msg_to_pub.data = flat_data
                # Add layout info: boxes × 24
                self.msg_to_pub.layout.dim = [
                    MultiArrayDimension(label="boxes", size=len(boxes), stride=len(boxes) * 24),
                    MultiArrayDimension(label="coords", size=24, stride=24)
                ]
                # self.bbox_pub.publish(msg)
                
              
                Pub_obs.publish(self.msg_to_pub)
                rospy.loginfo("----Detected! Count: %d----", len(boxes))
            else:
                rospy.loginfo("----No obstacles in front.----")
                self.if_obs.data[0]=0
            
            if side_obstacles:
                side_boxes = self.get_bounding_boxes(side_obstacles)
                if side_boxes:
                    self.if_obs.data[1] = 1
                else:
                    self.if_obs.data[1] = 0
            else:
                self.if_obs.data[1] = 0


            Pub_just.publish(self.if_obs)

        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            rospy.logwarn("TF transform failed: %s", str(e))

    def run(self):
        rospy.spin()

    def get_bounding_boxes(self, points, distance_threshold=0.3, min_cluster_size=5):
        """
            Naive clustering and AABB corner extraction for point groups.

            :param points: list of (x, y, z)
            :param distance_threshold: max distance between points in a cluster
            :param min_cluster_size: minimum number of points to accept a cluster
            :return: list of bounding boxes, each as a list of 8 corner points
        """
        if not points:
            return []

        # Convert to numpy array
        pts = np.array(points)
        clusters = []
        used = np.zeros(len(pts), dtype=bool)

        for i in range(len(pts)):
            if used[i]:
                continue
            cluster = [i]
            queue = [i]
            used[i] = True

            while queue:
                idx = queue.pop()
                dists = np.linalg.norm(pts - pts[idx], axis=1)
                neighbors = np.where((dists < distance_threshold) & (~used))[0]
                queue.extend(neighbors.tolist())
                cluster.extend(neighbors.tolist())
                used[neighbors] = True
                # find the points clusters
            if len(cluster) >= min_cluster_size:
                clusters.append(pts[cluster])

        # For each cluster, compute the 8 corner points of AABB
        bounding_boxes = []
        for cluster_pts in clusters:
            x_min, y_min, z_min = cluster_pts.min(axis=0)
            x_max, y_max, z_max = cluster_pts.max(axis=0)

            corners = [
                (x_min, y_min, z_min), (x_max, y_min, z_min),
                (x_max, y_max, z_min), (x_min, y_max, z_min),
                (x_min, y_min, z_max), (x_max, y_min, z_max),
                (x_max, y_max, z_max), (x_min, y_max, z_max)
            ]
            bounding_boxes.append(corners)

        return bounding_boxes



if __name__ == '__main__':
    Pub_obs = rospy.Publisher("/perception/obstacle_info", Float32MultiArray, queue_size=10)
    Pub_just = rospy.Publisher("/perception/obstacle_if_side",Int32MultiArray,queue_size=10)
    detector = ObstacleDetector()
    detector.run()
