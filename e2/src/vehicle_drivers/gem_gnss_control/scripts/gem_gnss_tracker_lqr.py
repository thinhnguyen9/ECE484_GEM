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

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt


class PurePursuit(object):
    
    def __init__(self):

        self.rate       = rospy.Rate(30)    # Thinh
        self.start_time = rospy.get_time()
        self.last_time  = self.start_time
        self.logtime    = 30.0      # seconds of data to log
        # self.logname    = str(self.start_time) + "_LQR_control_" + str(int(self.logtime)) + "sec.npy"
        self.logname    = "ActualRun_0512_LQR_control_" + str(int(self.logtime)) + "sec.npy"
        self.logdata    = []        # [time, x, u]
        self.logdone    = False

        self.look_ahead = 0.                                # TODO May 12
        self.r          = 0.3  # radius to look around, depends on the distance between waypoints
        self.wheelbase  = 1.75 # meters
        self.offset     = 0.46 # meters
        self.vref       = 1.5  # m/s, reference speed       # TODO May 12

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
            throttleLimits = (.29, .37),                    # vref = 1.5    # TODO May 12
            # throttleLimits = (.4, .6),                      # vref = 3.     # TODO May 12
            throttleRateLimits = (-.1, .1),                 # TODO: tune
            brakeLimits = (.0, .5),
            brakeRateLimits = (-5., .5)
        )
        self.MPC_horizon = 1        # optimize every ... steps
        self.linearize_method = 1   # linearize the system around:      # TODO May 12
                                    # 0 - velocity only (safer)
                                    # 1 - full states (more optimized, less stable)
        self.carLQR = LQR(n=self.GEM.n-1, m=self.GEM.m)
        # TODO: tune
        maxY = .3                   # max allowable cross-track error       # TODO May 12
        maxTheta = np.pi*10/180      # max allowable heading error          # TODO May 12
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
        V = np.diag([1e-3, 1e-3])       # measurement noise covariance
        W = np.diag([1e-1, 2e-1])       # process noise covariance - TODO: tune
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

        # read waypoints into the system 
        self.read_waypoints() 
        self.path_points_x = np.array(self.path_points_lon_x)
        self.path_points_y = np.array(self.path_points_lat_y)

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

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr

    def read_waypoints(self):
        # read recorded GPS lat, lon, heading
        dirname  = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../waypoints/xyhead_demo_pp.csv')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
        # x towards East and y towards North
        self.path_points_lon_x   = [float(point[0]) for point in path_points] # longitude
        self.path_points_lat_y   = [float(point[1]) for point in path_points] # latitude
        self.path_points_heading = [float(point[2]) for point in path_points] # heading
        self.wp_size             = len(self.path_points_lon_x)
        self.dist_arr            = np.zeros(self.wp_size)

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
            
            # self.path_points_x = np.array(self.path_points_lon_x)
            # self.path_points_y = np.array(self.path_points_lat_y)

            curr_x, curr_y, curr_yaw = self.get_gem_state()
            x0 = np.array([curr_x, curr_y, curr_yaw, self.delta, self.speed])

            # ----------------- Find waypoints -----------------
            # finding the distance of each way point from the current position
            for i in range(self.wp_size):
                self.dist_arr[i] = self.tools.point_point_distance((self.path_points_x[i], self.path_points_y[i]), x0[:2])

            # ----------------- Actual errors (for logging) -----------------
            # 0.3m within (curr_x, curr_y)
            r = 0.2
            idxList = []
            while len(idxList) < 2 and r < 5:   # need at least 2 points to do line fit
                r += 0.1
                idxList = np.where(self.dist_arr < r)[0]
            # print(str(len(idxList)) + " waypoints for r=" + str(r))
            if len(idxList) >= 2:
                wpList = [(self.path_points_x[i], self.path_points_y[i]) for i in idxList]
                ct_err_actual, hd_err_actual = self.tools.ErrorsFromWaypoints(x0[:3], wpList)
                ct_err_actual = xref[1] - ct_err_actual
                hd_err_actual = xref[2] - hd_err_actual
            else:
                print('Wtf no wp found ??')
                ct_err_actual, hd_err_actual = None, None

            # ----------------- Controller errors -----------------
            # Calculate errors used for control - with lookahead
            r = 0.2
            idxList = []
            while len(idxList) < 2 and r < 5:   # need at least 2 points to do line fit
                r += 0.1
                idxList = np.where( (self.dist_arr < self.look_ahead + r) & (self.dist_arr > self.look_ahead - r) )[0]
            if len(idxList) >= 2:
                wpList = [(self.path_points_x[i], self.path_points_y[i]) for i in idxList]
                ct_err, hd_err = self.tools.ErrorsFromWaypoints(x0[:3], wpList)
            else:
                ct_err, hd_err = 0.0, 0.0

            # ----------------- Calculate control -----------------
            # # Bound errors for safety       # TODO May 12
            # ct_err = max(-.5, min(.5, ct_err))
            # hd_err = max(-np.pi/6, min(np.pi/6, hd_err))

            xbar = np.array([0, ct_err, hd_err, x0[3], x0[4]])
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

            # ----------------- Kalman filter -----------------
            # Estimate y, theta with low-frequency measurements

            # Current estimation (x_e at time t)
            x_e += dx_e*dt

            # Current derivative (for x_e at time t+1): dx = Ax + Bu + L(y-Cx)
            meas_update = np.zeros(2)
            if time_step % 10 == 0: # new measurements available
                if ct_err_actual is None or hd_err_actual is None:
                    y_meas = 0.
                    theta_meas = 0.
                else:
                    y_meas = xref[1] - ct_err_actual
                    theta_meas = xref[2] - hd_err_actual
                y = np.array([y_meas, theta_meas])
                meas_update = self.KF.K.T @ (y - x_e)
            dx_e[0] = x0[4]*sin(x_e[1]) + meas_update[0]    # dy = v*sin(theta)
            dx_e[1] = x0[4]/self.wheelbase*tan(x0[3]) + meas_update[1]  # dtheta = v/L*tan(delta)

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
                    entry = [ current_time,
                              x0[0], x0[1], x0[2], x0[3], x0[4],
                              u[0], u[1], u[2],
                              ct_err_actual, hd_err_actual,
                              xref[1]-x_e[0], xref[2]-x_e[1] ]
                    self.logdata.append([float(x) if x is not None else 0.0 for x in entry])
                    # xhat.append([xref[1]-x_e[0], xref[2]-x_e[1]])
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
                        lane_x = self.path_points_x
                        lane_y = self.path_points_y
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


