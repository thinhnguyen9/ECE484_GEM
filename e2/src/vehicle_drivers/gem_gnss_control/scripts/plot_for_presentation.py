from control_utils import CarModel, LQR, Aux
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, tan, sqrt, atan2

GEM = CarModel(
    carLength = 1.75,
    steerSpeed = 2.5*35/630,
    approx_steerTau = 5.0,
    carAccel = 2.0,
    carDecel = -5.0,
    carDamp = 2.0/11.1,
    steerLimits = (-np.pi*35/180, np.pi*35/180),
    steerRateLimits = (-2.*35/630, 2.*35/630),
    throttleLimits = (0.2, 0.5),
    throttleRateLimits = (-1.0, .25),
    brakeLimits = (0.0, 1.0),
    brakeRateLimits = (-5.0, 5.0)
)

# =================================================
#              PP vs. LQR
# =================================================

filename_pp     = 'e2/src/vehicle_drivers/gem_gnss_control/scripts/ActualRun_0512_PP_control_300sec_v1.5.npy'
filename_lqr    = 'e2/src/vehicle_drivers/gem_gnss_control/scripts/ActualRun_0512_LQR_control_300sec_v1.5.npy'
# filename_lqr    = 'e2/src/vehicle_drivers/gem_gnss_control/scripts/ActualRun_0510_LQR_lanefollow_60sec_take2.npy'
vref = 1.5

with open(filename_pp, 'rb') as f:
    data = np.load(f)
    lane_x = np.load(f)
    lane_y = np.load(f)
tvec_pp = data[:,0] - data[0,0]
xvec_pp = data[:,1:6]
uvec_pp = data[:,6:9]
evec_pp = data[:,9:11]
# if xvec[:,3] is steering wheel (degree) instead of delta (rad)
for i in range(len(xvec_pp)):
    xvec_pp[i,3] = GEM.steer2delta(np.radians(xvec_pp[i,3]))
    uvec_pp[i,0] = GEM.steer2delta(np.radians(uvec_pp[i,0]))

with open(filename_lqr, 'rb') as f:
    data = np.load(f)
tvec_lqr = data[:,0] - data[0,0]
xvec_lqr = data[:,1:6]
uvec_lqr = data[:,6:9]
evec_lqr = data[:,9:11]
ehat_lqr = data[:,11:13]

# Calculate RMS errors
rms_tracking_error = np.sqrt(np.mean(evec_pp[:, 0]**2))  # Cross-track error
rms_heading_error = np.degrees(np.sqrt(np.mean(evec_pp[:, 1]**2)))   # Heading error
rms_velocity_error = np.sqrt(np.mean((xvec_pp[:, 4] - vref)**2))  # Velocity error
print(f"[PP] RMS Cross-track Error: {rms_tracking_error:.4f} m")
print(f"[PP] RMS Heading Error: {rms_heading_error:.4f} deg")
print(f"[PP] RMS Velocity Error: {rms_velocity_error:.4f} m/s")

rms_tracking_error = np.sqrt(np.mean(evec_lqr[:, 0]**2))  # Cross-track error
rms_heading_error = np.degrees(np.sqrt(np.mean(evec_lqr[:, 1]**2)))   # Heading error
rms_velocity_error = np.sqrt(np.mean((xvec_lqr[:, 4] - vref)**2))  # Velocity error
print(f"[LQR] RMS Cross-track Error: {rms_tracking_error:.4f} m")
print(f"[LQR] RMS Heading Error: {rms_heading_error:.4f} deg")
print(f"[LQR] RMS Velocity Error: {rms_velocity_error:.4f} m/s")


plt.subplot(2,2,1)
plt.plot(lane_x, lane_y, 'k--', lw=1, label='Desired')
plt.plot(xvec_pp[:,0], xvec_pp[:,1], 'r-', lw=1.5, label='PP')
plt.plot(xvec_lqr[:,0], xvec_lqr[:,1], 'b-', lw=1.5, label='LQR')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid()
plt.legend()
plt.title('2D path')
plt.axis('equal')

plt.subplot(2,2,2)
plt.axhline(y=vref, color='k', linestyle='--', linewidth=1, label='Desired')
plt.plot(tvec_pp, xvec_pp[:,4], 'r-', lw=1, label='PP')
plt.plot(tvec_lqr, xvec_lqr[:,4], 'b-', lw=1, label='LQR')
plt.xlabel('Time (s)')
plt.ylabel('m/s')
plt.grid()
plt.legend()
plt.title('Car velocity')

plt.show()


# =================================================
#              Kalman filter
# =================================================

filename = 'e2/src/vehicle_drivers/gem_gnss_control/scripts/ActualRun_0510_LQR_lanefollow_60sec_take1.npy'
# filename = 'e2/src/vehicle_drivers/gem_gnss_control/scripts/ActualRun_0510_LQR_lanefollow_60sec_03.npy'

with open(filename, 'rb') as f:
    data = np.load(f)
    lane_x = np.load(f)
    lane_y = np.load(f)
tvec = data[:,0] - data[0,0]
xvec = data[:,1:6]
uvec = data[:,6:9]
evec = data[:,9:11]
ehat = data[:,11:13]

# Lane from Kalman filter
cam2rear = .75
lane = []
for i in range(len(xvec)):
    goal_meas = np.array([0, -evec[i,0], 1])
    goal_est = np.array([0, -ehat[i,0], 1])
    th = xvec[i,2]
    T = np.array([[cos(th), -sin(th), xvec[i,0]],
                    [sin(th), cos(th), xvec[i,1]],
                    [0, 0, 1]])
    # endgoal in world frame (map)
    lane_meas = T @ goal_meas
    lane_est = T @ goal_est
    lane.append([lane_meas[0], lane_meas[1], 
                 lane_est[0], lane_est[1]])
lane = np.array(lane)


plt.subplot(2,2,3)
# plt.plot(lane_x, lane_y, 'k--', lw=.7, label='Measurement')
plt.plot(lane[:,0], lane[:,1], 'k--', lw=.7, label='Measurement')
plt.plot(lane[:,2], lane[:,3], 'r-', lw=1.5, label='Estimation')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid()
plt.legend()
plt.title('2D path')
plt.axis('equal')

plt.show()

