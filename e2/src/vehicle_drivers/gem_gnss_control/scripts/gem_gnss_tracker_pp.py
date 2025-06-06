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
        self.logtime    = 300.0      # seconds of data to log
        # self.logname    = str(self.start_time) + "_PP_control_" + str(int(self.logtime)) + "sec.npy"
        self.logname    = "ActualRun_0512_PP_control_" + str(int(self.logtime)) + "sec.npy"
        self.logdata    = []        # [time, x, u]
        self.logdone    = False
        self.tools      = Aux()

        self.look_ahead = 4
        self.wheelbase  = 1.75 # meters
        self.offset     = 0.46 # meters

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
        self.steer      = 0.0

        # read waypoints into the system 
        self.goal       = 0            
        self.read_waypoints() 

        self.desired_speed = 1.5  # m/s, reference speed    # TODO May 12
        self.max_accel     = 0.48 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

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
        self.steer_cmd.angular_velocity_limit = 4.0 # radians/second


    # def inspva_callback(self, inspva_msg):
    #     self.lat     = inspva_msg.latitude  # latitude
    #     self.lon     = inspva_msg.longitude # longitude
    #     self.heading = inspva_msg.azimuth   # heading in degrees
    
    def ins_callback(self, msg):
        self.heading = round(msg.heading, 6)
    
    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)


    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    def steer_callback(self, msg):
        self.steer = round(np.degrees(msg.manual_input),1)

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr

    def front2steer(self, f_angle):
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0
        return steer_angle

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
        # lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)

        # latrad = self.olat*np.pi/180.0 
        # mdeglon = 111415.13*np.cos(latrad) - 94.55*np.cos(3.0*latrad) + 0.12*np.cos(5.0*latrad)
        # mdeglat = 111132.09 - 566.05*np.cos(2.0*latrad) + 1.20*np.cos(4.0*latrad) - 0.002*np.cos(6.0*latrad)

        # lon_wp_x = (lon_wp - self.olon) * mdeglon
        # lat_wp_y = (lat_wp - self.olat) * mdeglat

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
        curr_x = local_x_curr - self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr - self.offset * np.sin(curr_yaw)

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

    # find the angle bewtween two vectors    
    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)

    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    def start_pp(self):

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


            self.path_points_x = np.array(self.path_points_lon_x)
            self.path_points_y = np.array(self.path_points_lat_y)

            curr_x, curr_y, curr_yaw = self.get_gem_state()

            # finding the distance of each way point from the current position
            for i in range(len(self.path_points_x)):
                self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y))
            
            
            # ====================================================================================================
            # Calculate actual errors - 0.3m within (curr_x, curr_y)
            r = 0.2
            idxList = []
            while len(idxList) < 2 and r < 5:   # need at least 2 points to do line fit
                r += 0.1
                idxList = np.where(self.dist_arr < r)[0]
            # print(str(len(idxList)) + " waypoints for r=" + str(r))
            if len(idxList) >= 2:
                wpList = [(self.path_points_x[i], self.path_points_y[i]) for i in idxList]
                ct_err_actual, hd_err_actual = self.tools.ErrorsFromWaypoints((curr_x, curr_y, curr_yaw), wpList)
                ct_err_actual = 0 - ct_err_actual
                hd_err_actual = 0 - hd_err_actual
            else:
                print('Wtf no wp found ??')
                ct_err_actual, hd_err_actual = None, None
            # ====================================================================================================


            # finding those points which are less than the look ahead distance (will be behind and ahead of the vehicle)
            goal_arr = np.where( (self.dist_arr < self.look_ahead + 0.3) & (self.dist_arr > self.look_ahead - 0.3) )[0]

            # finding the goal point which is the last in the set of points less than the lookahead distance
            for idx in goal_arr:
                v1 = [self.path_points_x[idx]-curr_x , self.path_points_y[idx]-curr_y]
                v2 = [np.cos(curr_yaw), np.sin(curr_yaw)]
                temp_angle = self.find_angle(v1,v2)
                # find correct look-ahead point by using heading information
                if abs(temp_angle) < np.pi/2:
                    self.goal = idx
                    break

            # finding the distance between the goal point and the vehicle
            # true look-ahead distance between a waypoint and current position
            L = self.dist_arr[self.goal]

            # find the curvature and the angle 
            alpha = self.heading_to_yaw(self.path_points_heading[self.goal]) - curr_yaw

            # ----------------- tuning this part as needed -----------------
            k       = 0.41 
            angle_i = math.atan((k * 2 * self.wheelbase * math.sin(alpha)) / L) 
            angle   = angle_i*2
            # ----------------- tuning this part as needed -----------------

            f_delta = round(np.clip(angle, -0.61, 0.61), 3)

            f_delta_deg = np.degrees(f_delta)

            # steering_angle in degrees
            steering_angle = self.front2steer(f_delta_deg)

            # if(self.gem_enable == True):
            #     print("Current index: " + str(self.goal))
            #     print("Forward velocity: " + str(self.speed))
            #     ct_error = round(np.sin(alpha) * L, 3)
            #     print("Crosstrack Error: " + str(ct_error))
            #     print("Front steering angle: " + str(np.degrees(f_delta)) + " degrees")
            #     print("Steering wheel angle: " + str(steering_angle) + " degrees" )
            #     print("\n")

            current_time = rospy.get_time()
            filt_vel     = self.speed_filter.get_data(self.speed)
            output_accel = self.pid_speed.get_control(current_time, self.desired_speed - filt_vel)

            if output_accel > self.max_accel:
                output_accel = self.max_accel

            if output_accel < 0.3:
                output_accel = 0.3

            if (f_delta_deg <= 30 and f_delta_deg >= -30):
                self.turn_cmd.ui16_cmd = 1
            elif(f_delta_deg > 30):
                self.turn_cmd.ui16_cmd = 2 # turn left
            else:
                self.turn_cmd.ui16_cmd = 0 # turn right

            self.accel_cmd.f64_cmd = output_accel
            self.steer_cmd.angular_position = np.radians(steering_angle)
            self.accel_pub.publish(self.accel_cmd)
            self.steer_pub.publish(self.steer_cmd)
            self.turn_pub.publish(self.turn_cmd)

            # ====================================================================================================
            # ----------------- Log data -----------------
            if not self.logdone:
                if current_time - self.start_time <= self.logtime:
                    entry = [ current_time,
                              curr_x, curr_y, curr_yaw, self.steer, self.speed,
                              steering_angle, output_accel, 0.0,
                              ct_err_actual, hd_err_actual ]
                    self.logdata.append([float(x) if x is not None else 0.0 for x in entry])
                else:
                    # Debug: check problematic entries
                    problem_entries = []
                    for i, entry in enumerate(self.logdata):
                        if not isinstance(entry, list) or len(entry) != 11:
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


