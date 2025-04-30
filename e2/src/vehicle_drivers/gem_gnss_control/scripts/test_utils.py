from control_utils import CarModel, LQR, Aux
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, tan, sqrt, atan2

def car_dx(theta, delta, v, L=1.75):
    dx = np.zeros(3)
    dx[0] = v*cos(theta)
    dx[1] = v*sin(theta)
    dx[2] = v/L*tan(delta)
    return dx

# =================================================
#       Front wheel and steering wheel angles
# =================================================
GEM = CarModel(
    carLength = 1.75,
    steerSpeed = 2.5*35/630,
    approx_steerTau = 5.0,
    carAccel = 2.0,
    carDecel = -5.0,
    carDamp = 2.0/11.1,
    steerLimits = (-np.pi*35/180, np.pi*35/180),
    throttleLimits = (0.2, 0.5),
    throttleRateLimits = (-1.0, .25),
    brakeLimits = (0.0, 1.0),
    brakeRateLimits = (-5.0, 5.0)
)

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
#                  Control utils
# =================================================
# tools = Aux()
# point = np.array([1535, -136])
# line = np.array([[1497, 15], [1507, 15]])
# d = tools.point_line_distance(point, line)
# print(d)


# =================================================
#              Plot experiment data
# =================================================
# with open('e2/src/vehicle_drivers/gem_gnss_control/scripts/pp_control.npy', 'rb') as f:
with open('e2/src/vehicle_drivers/gem_gnss_control/scripts/TEST_PP_control_140sec.npy', 'rb') as f:
# with open('e2/src/vehicle_drivers/gem_gnss_control/scripts/TEST_LQR_control_140sec.npy', 'rb') as f:
    data = np.load(f)
    lane_x = np.load(f)
    lane_y = np.load(f)
tvec = data[:,0] - data[0,0]
xvec = data[:,1:6]
uvec = data[:,6:9]
evec = data[:,9:11]

# if xvec[:,3] is steering wheel (degree) instead of delta (rad)
for i in range(len(xvec)):
    xvec[i,3] = GEM.steer2delta(np.radians(xvec[i,3]))
    uvec[i,0] = GEM.steer2delta(np.radians(uvec[i,0]))

# ------------- Kalman filter -------------
x_est = np.zeros((len(xvec), 3))    # x, y, theta
x_est[0,:] = xvec[0,0:3]
A,B = GEM.linearize(np.array([0, 0, 0, 0, 1.5]))
A = A[1:3,1:3]  # y, theta
C = np.array([[1., 0.], [0., 1.]])  # can measure y, theta
KF = LQR(n=2, m=2)
KF.setModel(A.T, C.T)
V = np.diag([1e-3, 1e-3])   # measurement noise covariance
W = np.diag([1., 1.])   # process noise covariance - TODO: tune
KF.setWeight(W, V)
KF.calculateGain()
update = np.zeros(3)
for i in range(len(xvec)):
    if i > 0:
        dt = tvec[i] - tvec[i-1]
        dx = car_dx(theta=x_est[i-1,2], delta=xvec[i-1,3], v=xvec[i-1,4], L=1.75)
        if i % 10 == 0:
            update[1:3] = KF.K.T @ (xvec[i-1,1:3] - x_est[i-1,1:3])
            dx += update
        x_est[i,0] = x_est[i-1,0] + dx[0]*dt
        x_est[i,1] = x_est[i-1,1] + dx[1]*dt
        x_est[i,2] = x_est[i-1,2] + dx[2]*dt

plt.subplot(2,2,1)
plt.plot(lane_x, lane_y, 'k--', lw=1, label='lane')
plt.plot(xvec[:,0], xvec[:,1], 'r-', lw=1.5, label='car')
plt.plot(x_est[:,0], x_est[:,1], 'm--', lw=1, label='est')
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
plt.plot(tvec, uvec[:,0]*180/np.pi, 'k--', lw=.7, label='command')
plt.plot(tvec, xvec[:,3]*180/np.pi, 'k-', lw=1, label='actual')
plt.plot(tvec, xvec[:,2]*180/np.pi, 'b-', lw=1, label='heading')
plt.xlabel('Time (s)')
plt.ylabel('deg')
plt.grid()
plt.legend()
plt.title('Steering angle')

plt.subplot(2,2,4)
# plt.plot(tvec[:i], xrefvec[:i,4], label='desired', c='black', ls='--', lw=.7)
plt.plot(tvec, xvec[:,4], 'k-', lw=1, label='actual')
plt.plot(tvec, uvec[:,1], 'b--', lw=1, label='throttle')
plt.plot(tvec, uvec[:,2], 'r--', lw=1, label='brake')
plt.xlabel('Time (s)')
plt.ylabel('m/s')
plt.grid()
plt.legend()
plt.title('Car velocity')

plt.show()

