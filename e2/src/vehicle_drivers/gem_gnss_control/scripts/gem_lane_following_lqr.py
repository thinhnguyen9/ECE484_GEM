#!/usr/bin/env python3

#================================================================
# File name: gem_gnss_pp_tracker_pid.py                                                                  
# Description: gnss waypoints tracker using pid and pure pursuit                                                                
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 08/02/2021                                                                
# Date last modified: 03/14/2025                                                
# Version: 1.0                                                                   
# Usage: rosrun gem_gnss gem_gnss_pp_tracker.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

from filters import OnlineFilter
from pid_controllers import PID
from control_utils import CarModel, LQR, Aux    # Thinh
from math import sin, cos, sqrt, tan
import matplotlib.pyplot as plt

# ROS Headers
# import alvinxy.alvinxy as axy # Import AlvinXY transformation module
import rospy

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64
# from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt


class PurePursuit(object):
    
    def __init__(self):

        self.rate       = rospy.Rate(30)    # Thinh
        self.start_time = rospy.get_time()
        self.last_time  = self.start_time
        self.logtime    = 60.0      # seconds of data to log
        # self.logname    = str(self.start_time) + "_LQR_control_" + str(int(self.logtime)) + "sec.npy"
        self.logname    = "TEST_LQR_lanefollow_" + str(int(self.logtime)) + "sec.npy"
        self.logdata    = []        # [time, x, u]
        self.logdone    = False

        self.look_ahead = 2.
        self.r          = 0.3  # radius to look around, depends on the distance between waypoints
        self.wheelbase  = 1.75 # meters
        self.offset     = 0.46 # meters
        self.cam2rear   = 0.75 # distance between front camera and rear wheel axle
        self.vref       = 1.5  # m/s, reference speed

        self.img_w      = 1280.
        self.img_h      = 720.
        self.p2m        = 4./700    # convert pixel -> meters (depends on lane detection unit)
        self.img_T      = np.array([[1, 0, 0],
                                    [0, -1, self.img_h],
                                    [0, 0, 1]])

        # -------------------- Controller setup --------------------
        self.tools = Aux()
        self.GEM = CarModel(
            carLength = self.wheelbase,
            steerSpeed = 2.5*35/630,
            approx_steerTau = 5.0,
            carAccel = 2.0,
            carDecel = -5.0,
            carDamp = 2.0/11.1,
            steerLimits = (-np.pi*35/180, np.pi*35/180),
            steerRateLimits = (-4.*35/630, 4.*35/630),      # TODO: tune
            throttleLimits = (.3, .4),                      # TODO: tune
            throttleRateLimits = (-.1, .1),                 # TODO: tune
            brakeLimits = (.0, .5),
            brakeRateLimits = (-5., .5)
        )
        self.MPC_horizon = 1        # optimize every ... steps
        self.linearize_method = 0   # linearize the system around:
                                    # 0 - velocity only (safer)
                                    # 1 - full states (more optimized, less stable)
        self.carLQR = LQR(n=self.GEM.n-1, m=self.GEM.m)
        # TODO: tune
        maxY = .3                   # max allowable cross-track error
        maxTheta = np.pi*10/180      # max allowable heading error
        maxDelta = np.pi*10/180      # max allowable steering angle error
        maxV = .1                   # max allowable velocity error
        Q = np.diag([   1/(maxY**2),
                        1/(maxTheta**2),
                        1/(maxDelta**2),
                        1/(maxV**2) ])   # [y, theta, delta, v] - x is removed
        R = np.diag([   1/(self.GEM.delta_max**2),
                        1/(self.GEM.throttle_max**2),
                        20/(self.GEM.brake_max**2) ])
        self.carLQR.setWeight(Q, R)
        # -------------------- Kalman filter --------------------
        A,B = self.GEM.linearize(np.array([0, 0, 0, 0, self.vref]))
        A = A[1:3,1:3]  # y, theta
        C = np.array([[1., 0.], [0., 1.]])  # can measure y, theta
        self.KF = LQR(n=2, m=2)
        self.KF.setModel(A.T, C.T)
        V = np.diag([1e-2, 1e-2])       # measurement noise covariance
        W = np.diag([1e-1, 1e-1])       # process noise covariance - TODO: tune
        self.KF.setWeight(W, V)
        self.KF.calculateGain()

        # -------------------- ROS setup --------------------
        # self.gnss_sub_old   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        # we replaced novatel hardware with septentrio hardware on e2
        self.gnss_sub   = rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, self.gnss_callback)
        self.ins_sub    = rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, self.ins_callback)
        self.lat        = 0.0
        self.lon        = 0.0
        self.heading    = 0.0

        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0

        self.olat       = 40.0928563
        self.olon       = -88.2359994

        self.steer_sub  = rospy.Subscriber("/pacmod/parsed_tx/steer_rpt", SystemRptFloat, self.steer_callback)
        self.steer      = 0.0   # steering wheel angle, rad
        self.delta      = 0.0   # front wheel angle, rad

        # # read waypoints into the system 
        # self.read_waypoints() 
        # self.path_points_x = np.array(self.path_points_lon_x)
        # self.path_points_y = np.array(self.path_points_lat_y)

        # Subscribe to lane detection output
        self.endgoal_sub = rospy.Subscriber("/lane_detection/endgoal", PoseStamped, self.endgoal_callback)
        self.waypoints_sub = rospy.Subscriber("/lane_detection/waypoints", Path, self.waypoints_callback)
        self.endgoal = np.zeros(2)  # [x, y]
        self.waypoints = []         # [[x1,y1], [x2,y2], ...] at current time t
        self.lane = []              # [[x1,y1], [x2,y2], ...] (endgoal) for time 0...T
        self.new_waypoints_flag = False

        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = False

        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 4.0 # radians/second    # TODO: tune
    
    def ins_callback(self, msg):
        self.heading = round(msg.heading, 6)
    
    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)

    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    def steer_callback(self, msg):
        self.steer = round(msg.manual_input, 4)
        self.delta = round(self.GEM.steer2delta(msg.manual_input), 4)

    def endgoal_callback(self, msg):
        """
        Callback for lane detection endgoal messages.
        
        Updates target position for steering control based on lane detection.
        
        Args:
            msg: PoseStamped message containing target position in image
        """
        # self.endgoal[0] = msg.pose.position.x
        # self.endgoal[1] = msg.pose.position.y

        # endgoal in image coordinate
        p0 = np.array([msg.pose.position.x, msg.pose.position.y, 1])
        p1 = self.img_T @ p0
        self.endgoal[0] = p1[0]*self.p2m
        self.endgoal[1] = p1[1]*self.p2m

    def waypoints_callback(self, msg):
        if len(msg.poses) > 0:
            self.waypoints = []
            for posestamp in msg.poses:
                wp0 = np.array([posestamp.pose.position.x, posestamp.pose.position.y, 1])
                wp1 = self.img_T @ wp0
                self.waypoints.append(wp1[:2]*self.p2m)
            self.new_waypoints_flag = True

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr

    def wps_to_local_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = self.tools.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        return lon_wp_x, lat_wp_y

    def get_gem_state(self):

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # reference point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps_to_local_xy(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = self.heading_to_yaw(self.heading) 

        # reference point is located at the center of rear axle
        curr_x = local_x_curr - self.offset * cos(curr_yaw)
        curr_y = local_y_curr - self.offset * sin(curr_yaw)

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

    def start_pp(self):

        # Initialize
        # curr_x, curr_y, curr_yaw = self.get_gem_state()
        # x0 = np.array([curr_x, curr_y, curr_yaw, self.delta, self.speed])   # [x; y; theta; delta (steering angle); v (car speed)]
        u0 = np.zeros(self.GEM.m)   # [delta_des; throttle; brake]
        # u = np.zeros(self.GEM.m)
        xref = np.array([0.0, 0.0, 0.0, 0.0, self.vref])
        time_step = 0

        # For Kalman filter
        x_e = np.zeros(2)   # initial estimate [y, theta]
        dx_e = np.zeros(2)  # initial estimate derivatives
        # xhat = []   # estimated ct_err and hd_err

        # Wait for GNSS data and waypoints to be loaded
        rospy.sleep(1.0)
        
        while not rospy.is_shutdown():

            if (self.gem_enable == False):

                if(self.pacmod_enable == True):

                    # ---------- enable PACMod ----------

                    # enable forward gear
                    self.gear_cmd.ui16_cmd = 3

                    # enable brake
                    self.brake_cmd.enable  = True
                    self.brake_cmd.clear   = False
                    self.brake_cmd.ignore  = False
                    self.brake_cmd.f64_cmd = 0.0

                    # enable gas 
                    self.accel_cmd.enable  = True
                    self.accel_cmd.clear   = False
                    self.accel_cmd.ignore  = False
                    self.accel_cmd.f64_cmd = 0.0

                    self.gear_pub.publish(self.gear_cmd)
                    print("Foward Engaged!")

                    self.turn_pub.publish(self.turn_cmd)
                    print("Turn Signal Ready!")
                    
                    self.brake_pub.publish(self.brake_cmd)
                    print("Brake Engaged!")

                    self.accel_pub.publish(self.accel_cmd)
                    print("Gas Engaged!")

                    self.gem_enable = True

            # ====================================================================================================
            
            # ----------------- Get current state -----------------
            current_time = rospy.get_time()
            dt = current_time - self.last_time
            self.last_time = current_time

            curr_x, curr_y, curr_yaw = self.get_gem_state()
            x0 = np.array([curr_x, curr_y, curr_yaw, self.delta, self.speed])

            # ----------------- Calculate errors -----------------
            if len(self.waypoints) >= 2:
                ct_err, hd_err = self.tools.ErrorsFromWaypoints([self.img_w/2, 0, np.pi/2], self.waypoints)  # self.waypoints are in car coordinates
            else:
                ct_err, hd_err = 0.0, 0.0

            # ----------------- Kalman filter -----------------
            # Estimate y, theta with low-frequency measurements

            # Current estimation (x_e at time t)
            x_e += dx_e*dt

            # Current derivative (for x_e at time t+1): dx = Ax + Bu + L(y-Cx)
            meas_update = np.zeros(2)
            if self.new_waypoints_flag: # new measurements available
                y_meas = xref[1] - ct_err
                theta_meas = xref[2] - hd_err
                y = np.array([y_meas, theta_meas])
                meas_update = self.KF.K.T @ (y - x_e)
                self.new_waypoints_flag = False
            dx_e[0] = x0[4]*sin(x_e[1]) + meas_update[0]    # dy = v*sin(theta)
            dx_e[1] = x0[4]/self.wheelbase*tan(x0[3]) + meas_update[1]  # dtheta = v/L*tan(delta)

            # ----------------- Calculate control -----------------
            xbar = np.array([0, x_e[0], x_e[1], x0[3], x0[4]])
            if time_step % self.MPC_horizon == 0:
                if self.linearize_method == 0:
                    A, B = self.GEM.linearize(np.array([0, 0, 0, 0, x0[4]]))
                elif self.linearize_method == 1:
                    A, B = self.GEM.linearize(xbar)
                self.carLQR.setModel(A[1:,1:], B[1:,:])
                self.carLQR.calculateGain()
                K = np.hstack((np.zeros((self.GEM.m, 1)), self.carLQR.K))
            u = K.dot(xref.T - xbar.T)
            u = self.GEM.saturateControl(u, u0, dt)
            # only change throttle if speed error is significant - prevent oscillation
            if abs(self.vref - self.speed) < 0.1:
                u[1] = u0[1]
            u0 = u
            time_step += 1

            # ----------------- Publish control -----------------
            # if (f_delta_deg <= 30 and f_delta_deg >= -30):
            #     self.turn_cmd.ui16_cmd = 1
            # elif(f_delta_deg > 30):
            #     self.turn_cmd.ui16_cmd = 2 # turn left
            # else:
            #     self.turn_cmd.ui16_cmd = 0 # turn right

            self.steer_cmd.angular_position = self.GEM.delta2steer(u[0])
            self.accel_cmd.f64_cmd = u[1]
            self.brake_cmd.f64_cmd = u[2]
            
            self.accel_pub.publish(self.accel_cmd)
            self.steer_pub.publish(self.steer_cmd)
            self.turn_pub.publish(self.turn_cmd)
            self.brake_pub.publish(self.brake_cmd)  # Thinh

            # ----------------- Log data -----------------
            if not self.logdone:
                if current_time - self.start_time <= self.logtime:

                    # Lane position in world frame (detected by camera)
                    th = -np.pi/2 + x0[2]
                    T = np.array([[cos(th), -sin(th), x0[0]],
                                  [sin(th), cos(th), x0[1]],
                                  [0, 0, 1]])
                    p1 = np.array([self.endgoal[0], self.endgoal[1]+self.cam2rear, 1])    # endgoal in car frame (x+: to the right; y+: to the front)
                    p0 = T @ p1     # endgoal in world frame (map)
                    self.lane.append([p0[0], p0[1]])

                    entry = [ current_time,
                              x0[0], x0[1], x0[2], x0[3], x0[4],
                              u[0], u[1], u[2],
                              ct_err, hd_err,
                              xref[1]-x_e[0], xref[2]-x_e[1] ]
                    self.logdata.append([float(x) if x is not None else 0.0 for x in entry])
                    
                else:
                    # Debug: check problematic entries
                    problem_entries = []
                    for i, entry in enumerate(self.logdata):
                        if not isinstance(entry, list) or len(entry) != 13:
                            problem_entries.append((i, entry))
                            
                    if problem_entries:
                        print(f"Found {len(problem_entries)} problematic entries:")
                        for i, entry in problem_entries[:5]:  # Show up to 5 examples
                            print(f"Index {i}: {entry}, type: {type(entry)}")
                            
                    # Try saving with manual conversion
                    try:
                        data_array = np.array(self.logdata, dtype=float)
                        lane = np.array(self.lane, dtype=float)
                        lane_x = lane[:,0]
                        lane_y = lane[:,1]
                        with open(self.logname, 'wb') as f:
                            np.save(f, data_array)
                            np.save(f, lane_x)
                            np.save(f, lane_y)
                        print("Data logged into " + self.logname)
                    except Exception as e:
                        print(f"Error during save: {e}")
                    
                    self.logdone = True
                    # ----------------- Plot here cause i'm lazy -----------------
                    # xhat = np.array(xhat)
                    # plt.subplot(1,2,1)
                    # plt.plot(data_array[:,0], data_array[:,9], 'k--', lw=1, label='actual')
                    # plt.plot(data_array[:,0], xhat[:,0], 'k-', lw=1, label='estimated')
                    # plt.xlabel('Time (s)')
                    # plt.ylabel('m')
                    # plt.grid()
                    # plt.legend()
                    # plt.title('Cross-track error')

                    # plt.subplot(1,2,2)
                    # plt.plot(data_array[:,0], data_array[:,10], 'k--', lw=1, label='actual')
                    # plt.plot(data_array[:,0], xhat[:,1], 'k-', lw=1, label='estimated')
                    # plt.xlabel('Time (s)')
                    # plt.ylabel('rad')
                    # plt.grid()
                    # plt.legend()
                    # plt.title('Heading error')

                    # self.lane = np.array(self.lane)
                    # plt.plot(self.lane[:,0], self.lane[:,1], 'k--', lw=1, label='lane')
                    # plt.plot(self.lane[:,2], self.lane[:,3], 'r-', lw=1.5, label='car')
                    # plt.xlabel('X (m)')
                    # plt.ylabel('Y (m)')
                    # plt.grid()
                    # plt.legend()
                    # plt.title('2D path')
                    # plt.axis('equal')

                    # plt.show()
                    # ------------------------------------------------------------
                    break   # stop running when data is logged
            # ====================================================================================================

            self.rate.sleep()


def pure_pursuit():

    rospy.init_node('gnss_pp_node', anonymous=True)
    pp = PurePursuit()

    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()


