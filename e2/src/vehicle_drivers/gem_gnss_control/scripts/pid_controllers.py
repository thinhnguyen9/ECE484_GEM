#!/usr/bin/env python3

#================================================================
# File name: pure_pursuit_pid_tracker_controller.py                                                                  
# Description: gnss waypoints tracker using pid and pure pursuit                                                                
# Author: Hang Cui, Sidharth Anand, John Pohovey
# Email: hangcui3@illinois.edu, sanand12@illinois.edu, jpohov2@illinois.edu
# Date created: 08/02/2021
# Date last modified: 03/14/2025
# Version: 1.1                                                  
# Usage: pure_pursuit_pid_tracker_controller.py
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import time
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
import rospy
# import alvinxy.alvinxy as axy # Import AlvinXY transformation module
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64
# from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt


class PID(object):
    """
    Generic PID controller implementation with anti-windup protection.
    
    This class provides a flexible PID controller that can be used for
    various control applications including steering and speed control.
    
    Attributes:
        kp (float): Proportional gain
        ki (float): Integral gain
        kd (float): Derivative gain
        wg (float): Windup guard limit (anti-windup)
    """
    def __init__(self, kp, ki, kd, wg=None):
        """
        Initialize PID controller with gains and optional windup guard.
        
        Args:
            kp: Proportional gain - immediate response to error
            ki: Integral gain - compensates for steady-state error
            kd: Derivative gain - provides damping effect
            wg: Windup guard limit - prevents integral term from growing too large
        """
        self.iterm = 0            # Integral term accumulator
        self.last_t = None        # Time of last control computation
        self.last_e = 0           # Previous error value
        self.kp = kp              # Proportional gain
        self.ki = ki              # Integral gain
        self.kd = kd              # Derivative gain
        self.wg = wg              # Windup guard limit
        self.derror = 0           # Derivative of error (rate of change)

    def reset(self):
        """
        Reset controller state.
        
        Call this function when restarting control or changing setpoints
        significantly to prevent accumulated integral term from causing
        undesired behavior.
        """
        self.iterm = 0        # Clear integral accumulator
        self.last_e = 0       # Reset previous error
        self.last_t = None    # Reset timing information

    def get_control(self, t, e, fwd=0):
        """
        Calculate control output based on error and time.
        
        Implements standard PID control algorithm:
        u(t) = Kp*e(t) + Ki∫e(τ)dτ + Kd*de/dt + feedforward
        
        Args:
            t: Current time in seconds
            e: Current error value (setpoint - measured value)
            fwd: Optional feedforward term
            
        Returns:
            Control output value
        """
        # Handle first iteration (no derivative)
        if self.last_t is None:
            self.last_t = t
            de = 0
        else:
            # Calculate error derivative (rate of change)
            de = (e - self.last_e) / (t - self.last_t)

        # Outlier detection - if error jumps too much, ignore derivative term
        # This prevents large control actions due to sensor noise or outliers
        if abs(e - self.last_e) > 0.5:
            de = 0

        # Update integral term (∫e(τ)dτ)
        self.iterm += e * (t - self.last_t)

        # Apply anti-windup guard to limit integral term
        if self.wg is not None:
            if self.iterm > self.wg:
                self.iterm = self.wg        # Cap positive integral term
            elif self.iterm < -self.wg:
                self.iterm = -self.wg       # Cap negative integral term

        # Store current values for next iteration
        self.last_e = e
        self.last_t = t
        self.derror = de

        # Return PID control output + feedforward term
        return fwd + self.kp * e + self.ki * self.iterm + self.kd * de
