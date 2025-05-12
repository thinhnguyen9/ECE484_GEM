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

filename_pp     = 'e2/src/vehicle_drivers/gem_gnss_control/scripts/ActualRun_0501_PP_control_200sec.npy'
filename_lqr    = 'e2/src/vehicle_drivers/gem_gnss_control/scripts/ActualRun_0505_LQR_control_150sec.npy'
# filename_lqr    = 'e2/src/vehicle_drivers/gem_gnss_control/scripts/ActualRun_0510_LQR_lanefollow_60sec_take2.npy'

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


plt.subplot(2,2,1)
plt.plot(lane_x, lane_y, 'k--', lw=1, label='Waypoints')
plt.plot(xvec_pp[:,0], xvec_pp[:,1], 'r-', lw=1.5, label='PP')
plt.plot(xvec_lqr[:,0], xvec_lqr[:,1], 'b-', lw=1.5, label='LQR')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid()
plt.legend()
plt.title('2D path')
plt.axis('equal')

t = 4000
plt.subplot(2,2,2)
plt.plot(tvec_pp[:t], xvec_pp[:t,4], 'k-', lw=1, label='PP')
plt.plot(tvec_lqr[:t], xvec_lqr[:t,4], 'b-', lw=1, label='LQR')
plt.xlabel('Time (s)')
plt.ylabel('m/s')
plt.grid()
plt.legend()
plt.title('Car velocity')

"""
plt.subplot(2,2,2)
plt.plot(tvec, evec[:,0], 'r--', lw=.5, label='cross-track err (m)')
plt.plot(tvec, evec[:,1], 'b--', lw=.5, label='heading err (rad)')
if file=='lqr':
    plt.plot(tvec, ehat[:,0], 'r-', lw=1.5, label='ct_est')
    plt.plot(tvec, ehat[:,1], 'b-', lw=1.5, label='hd_est')
plt.xlabel('Time (s)')
plt.ylabel('m, rad')
plt.grid()
plt.legend()
plt.title('Cross-track error')

plt.subplot(2,2,3)
plt.plot(tvec, uvec[:,0]*180/np.pi, 'k--', lw=.7, label='command')
plt.plot(tvec, xvec[:,3]*180/np.pi, 'k-', lw=1, label='actual')
plt.plot(tvec, xvec[:,2]*180/np.pi, 'b-', lw=1, label='heading')
plt.xlabel('Time (s)')
plt.ylabel('deg')
plt.grid()
plt.legend()
plt.title('Steering angle')


"""

plt.show()

