from control_utils import CarModel
import numpy as np
import matplotlib.pyplot as plt

# =================================================
#       Front wheel and steering wheel angles
# =================================================
# GEM = CarModel(
#     carLength = 1.75,
#     steerSpeed = 2.5*35/630,
#     approx_steerTau = 5.0,
#     carAccel = 2.0,
#     carDecel = -5.0,
#     carDamp = 2.0/11.1,
#     steerLimits = (-np.pi*35/180, np.pi*35/180),
#     throttleLimits = (0.2, 0.5),
#     throttleRateLimits = (-1.0, .25),
#     brakeLimits = (0.0, 1.0),
#     brakeRateLimits = (-5.0, 5.0)
# )

# delta1 = np.linspace(-1, 1, 100, endpoint=True)
# steer1 = np.array([0.0]*len(delta1))
# for i in range(len(delta1)):
#     steer1[i] = GEM.delta2steer(delta1[i])

# steer2 = np.linspace(-20, 20, 41, endpoint=True)
# delta2 = np.array([0.0]*len(steer2))
# for i in range(len(steer2)):
#     delta2[i] = GEM.steer2delta(steer2[i])

# plt.plot(delta1, steer1, label='delta2steer', c='black', lw=1)
# plt.plot(delta2, steer2, label='steer2delta', c='r', ls='', marker='.', lw=1)
# plt.xlabel('delta')
# plt.ylabel('steering wheel')
# plt.grid()
# plt.legend()
# plt.show()

# =================================================
#              Plot experiment data
# =================================================
# with open('e2\\src\\vehicle_drivers\\gem_gnss_control\\scripts\\test.npy', 'rb') as f:
with open('e2/src/vehicle_drivers/gem_gnss_control/scripts/pp_control.npy', 'rb') as f:
    data = np.load(f)
    lane_x = np.load(f)
    lane_y = np.load(f)
tvec = data[:,0] - data[0,0]
xvec = data[:,1:6]
uvec = data[:,6:9]
evec = data[:,9:11]

plt.subplot(2,2,1)
plt.plot(lane_x, lane_y, label='lane', c='black', ls='--', lw=1)
plt.plot(xvec[:,0], xvec[:,1], label='car', c='red', ls='-', lw=1.5)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid()
plt.legend()
plt.title('2D path')
plt.axis('equal')

plt.subplot(2,2,2)
plt.plot(tvec, evec[:,0], label='cross-track err (m)', lw=1)
plt.plot(tvec, evec[:,1], label='heading err (rad)', lw=1)
plt.xlabel('Time (s)')
plt.ylabel('m, rad')
plt.grid()
plt.legend()
plt.title('Cross-track error')

plt.subplot(2,2,3)
plt.plot(tvec, uvec[:,0]*180/np.pi, label='command', lw=.7)
plt.plot(tvec, xvec[:,3]*180/np.pi, label='actual', c='red', lw=1)
plt.xlabel('Time (s)')
plt.ylabel('deg')
plt.grid()
plt.legend()
plt.title('Steering angle')

plt.subplot(2,2,4)
# plt.plot(tvec[:i], xrefvec[:i,4], label='desired', c='black', ls='--', lw=.7)
plt.plot(tvec, xvec[:,4], label='actual', c='black', ls='-', lw=1)
plt.plot(tvec, uvec[:,1], label='throttle', c='blue', ls='--', lw=1)
plt.plot(tvec, uvec[:,2], label='brake', c='red', ls='--', lw=1)
plt.xlabel('Time (s)')
plt.ylabel('m/s')
plt.grid()
plt.legend()
plt.title('Car velocity')

plt.show()