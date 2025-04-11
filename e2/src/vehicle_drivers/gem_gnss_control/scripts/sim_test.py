import numpy as np
from math import sin, cos, tan, atan2, sqrt
import matplotlib.pyplot as plt
from control_utils import CarModel, LQR, Aux
from sim_waypoints import WayPoints

# ================================
#           Car params
# ================================
'''
    carLength           : GEM e2: 1.75m, GEM e4: 2.56m
    steerSpeed          : rate of change for steering angle (rad/s)
    approx_steerTau     : approx. time constant (1st-order) of the steering angle (must be slower than actual)
    carAccel            : car acceleration for throttle = 1 (m/s^2)
    carDecel            : car deceleration for brake = 1 (m/s^2)
    carDamp             : carAccel/Vmax - damping to slow the car down (a = -v_damp*v)
    steerLimits         : min/max steering angle
    throttleLimits      : min/max throttle command
    throttleRateLimits  : min/max throttle command rate of change
    brakeLimits         : min/max brake command
'''
GEM = CarModel(
    carLength = 1.75,
    steerSpeed = 2.5*35/630,
    approx_steerTau = 5.0,
    carAccel = 2.0,
    carDecel = -5.0,
    carDamp = 2.0/11.1,
    steerLimits = (-np.pi*35/180, np.pi*35/180),
    throttleLimits = (0.0, 0.5),
    throttleRateLimits = (-1.0, .25),
    brakeLimits = (0.0, 1.0),
    brakeRateLimits = (-5.0, 5.0)
)


# ================================
#           Sim params
# ================================
'''
    Original system:
        x = [x_loc; y_loc; theta; delta (steering angle); v (car speed)]
        u = [delta_des; throttle; brake]
'''
T, ts = 200, 1/30
tvec = np.arange(0, T+ts, ts)
x0 = np.array([ 0.0,  # x
                1.0,  # y
                0.0,  # theta
                0.0,  # delta
                0.0   # v
            ])
xref = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
MPC_horizon = 1        # optimize every ... ts
addNoise = False
linearize_method = 1    # linearize the system around:
                        # 0 - velocity only (safer)
                        # 1 - full states (more optimized, less stable)

'''
    For LQR, x_loc (x[0]) is removed (since we control v, not x_loc)
        x = [   y (cross-track err);
                theta (heading err);
                delta (steering angle);
                v (car speed)   ]
        u = [delta_des; throttle; brake]
'''
continuousLQR = LQR(n=GEM.n-1, m=GEM.m)
# Tune these
maxY = .1                   # max allowable cross-track error
maxTheta = np.pi*5/180      # max allowable heading error
maxDelta = np.pi*5/180      # max allowable steering angle error
maxV = .1                   # max allowable velocity error
Q = np.diag([1/(maxY**2), 1/(maxTheta**2), 1/(maxDelta**2), 1/(maxV**2)])   # [y, theta, delta, v] - x is removed
R = np.diag([1/(GEM.delta_max**2), 1/(GEM.throttle_max**2), 100/(GEM.brake_max**2)])
continuousLQR.setWeight(Q, R)

# ================================
#           Waypoints
# ================================
wp = WayPoints()
map = 3
waypoints = wp.getWayPoints(option=map)


# ================================
#           Functions
# ================================
g = Aux()
def cross_track_distance(currLoc, wpList):
    '''
        wpList: [last wp, next wp]
    '''
    if map == 3:
        if currLoc[0] >= 0 and currLoc[0] <= 50: # straight path
            return g.point_line_distance(currLoc, wpList)
        elif currLoc[0] > 50:   # circle O=(50,-5), R=5
            return g.point_point_distance(currLoc, (50,-5)) - 5
        elif currLoc[0] < 0:   # circle O=(0,-5), R=5
            return g.point_point_distance(currLoc, (0,-5)) - 5
    else:
        return g.point_line_distance(currLoc, wpList)

def heading_wrt_track(currLoc, currTheta, wpList):
    path = wpList[1] - wpList[0]
    res = currTheta - atan2(path[1], path[0])

    # if map == 3:
    #     if currLoc[0] >= 0 and currLoc[0] <= 50: # straight path
    #         pass
    #     elif currLoc[0] > 50:   # circle O=(50,-5), R=5
    #         r = currLoc - (50,-5)
    #         res = currTheta - (atan2(r[1], r[0]) - np.pi/2)
    #     elif currLoc[0] < 0:   # circle O=(0,-5), R=5
    #         r = currLoc - (0,-5)
    #         res = currTheta - (atan2(r[1], r[0]) - np.pi/2)

    while res > np.pi:
        res = res - 2*np.pi
    while res < -np.pi:
        res = res + 2*np.pi
    return res


# ================================
#           Run sim
# ================================
N = np.size(tvec)
xrefvec = np.zeros((N, GEM.n))
xvec = np.zeros((N, GEM.n))
uvec = np.zeros((N, GEM.m))
evec = np.zeros((N, 2))     # [cross-track error, heading error]
wp_index = 0
u0 = np.zeros(GEM.m)
u = np.zeros(GEM.m)
# delta_filt = u0[0]

# add noise
GPS_accuracy = .05
accuracyX = sqrt(GPS_accuracy**2/2)
xyNoise = np.random.normal(0, accuracyX/3, size=(N,2))  # 3 standard deviations = max error

V_accuracy = .05
vNoise = np.random.normal(0, V_accuracy/3, size=(N,1))

theta_accuracy = .01
thetaNoise = np.random.normal(0, theta_accuracy/3, size=(N,1))

for i in range(N):

    if addNoise:
        x0[0] = x0[0] + xyNoise[i,0]
        x0[1] = x0[1] + xyNoise[i,1]
        x0[2] = x0[2] + thetaNoise[i]
        x0[4] = x0[4] + vNoise[i]

    if g.point_point_distance(x0[:2], waypoints[wp_index]) < 2:
        print('Reached waypoint number ' + str(wp_index) + '. Current pos: ' + str(x0[:2]) + '. Time: ' + str(tvec[i]))
        wp_index += 1
    if wp_index >= len(waypoints):
        break
    # if wp_index == 2:     # change theta_ref at curves doesn't help
    #     # xref[2] = -np.pi/4
    #     xref[4] = 1
    # if map == 3:
    #     if x0[0] < 10 or x0[0] > 40: # slow down at curves - for waypoints map 3
    #         xref[4] = 1.5
    #     else:
    #         xref[4] = 3.0
        # xref[4] = g.calculateVref(x0[:2], waypoints[wp_index:], lookahead=10, defaultV=(1, 5))
        
        # if x0[0] > 5 and x0[0] < 45:
        #     xref[1] = 1.0     # change lane
        # else:
        #     xref[1] = 0.0
    
    # # slow down when turning too aggressively
    # delta_filt = .99*delta_filt + .01*abs(uvec[i-1,0])
    # if delta_filt > np.pi*20/180:
    #     xref[4] = 1.0

    y, theta = g.ErrorsFromWaypoints(x0[:3], waypoints[wp_index-1 : wp_index+1])
    # y = cross_track_distance(x0[:2], waypoints[wp_index-1 : wp_index+1])
    # theta = heading_wrt_track(x0[:2], x0[2], waypoints[wp_index-1 : wp_index+1])
    evec[i] = [xref[1]-y, xref[2]-theta]
    xbar = np.array([0, y, theta, x0[3], x0[4]])

    if i % MPC_horizon == 0:
        # print(np.array([x0[2], theta])*180/np.pi)

        if linearize_method == 0:
            A, B = GEM.linearize(np.array([0, 0, 0, 0, x0[4]]))
        elif linearize_method == 1:
            A, B = GEM.linearize(xbar)
        continuousLQR.setModel(A[1:,1:], B[1:,:])
        continuousLQR.calculateGain()
        K = np.hstack((np.zeros((GEM.m, 1)), continuousLQR.K))

    if tvec[i] % 10 == 0:
        print('Gain at time t = ' + str(tvec[i]) + 's:')
        print(K)

    u = K.dot(xref.T - xbar.T)
    u = GEM.saturateControl(u, u0, ts)
    uvec[i] = u
    xvec[i] = x0
    xrefvec[i] = xref

    dx = GEM.dx(x0, u)
    x0 = x0 + dx*ts
    u0 = u


yRMSE = sqrt(np.mean(np.power(evec[:i,0], 2)))
thetaRMSE = sqrt(np.mean(np.power(evec[:i,1], 2)))
print('====================================\nFinished or timed out.\n====================================')
print('Cross-track RMSE: ' + str(yRMSE) + ' m')
print('Heading RMSE: ' + str(thetaRMSE) + ' rad')

with open('e2\\src\\vehicle_drivers\\gem_gnss_control\\scripts\\test.npy', 'wb') as f:
    np.save(f, xvec)
with open('e2\\src\\vehicle_drivers\\gem_gnss_control\\scripts\\test.npy', 'rb') as f:
    xvec_temp = np.load(f)

# ================================
#           Plot
# ================================
plt.subplot(2,2,1)
# plt.plot(lane[:,0], lane[:,1], label='lane')
plt.plot(waypoints[:,0], waypoints[:,1], label='lane', c='black', ls='--', lw=1)
plt.plot(xvec[:i,0], xvec[:i,1], label='car', c='red', ls='-', lw=1.5)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid()
plt.legend()
plt.title('2D path')
plt.axis('equal')

plt.subplot(2,2,2)
plt.plot(tvec[:i], evec[:i,0], label='cross-track err (m)', lw=1)
plt.plot(tvec[:i], evec[:i,1], label='heading err (rad)', lw=1)
plt.xlabel('Time (s)')
plt.ylabel('m, rad')
plt.grid()
plt.legend()
plt.title('Cross-track error')

plt.subplot(2,2,3)
plt.plot(tvec[:i], uvec[:i,0]*180/np.pi, label='command', lw=.7)
plt.plot(tvec[:i], xvec[:i,3]*180/np.pi, label='actual', c='red', lw=1)
plt.xlabel('Time (s)')
plt.ylabel('deg')
plt.grid()
plt.legend()
plt.title('Steering angle')

plt.subplot(2,2,4)
plt.plot(tvec[:i], xrefvec[:i,4], label='desired', c='black', ls='--', lw=.7)
plt.plot(tvec[:i], xvec[:i,4], label='actual', c='black', ls='-', lw=1)
plt.plot(tvec[:i], uvec[:i,1], label='throttle', c='blue', ls='--', lw=1)
plt.plot(tvec[:i], uvec[:i,2], label='brake', c='red', ls='--', lw=1)
plt.xlabel('Time (s)')
plt.ylabel('m/s')
plt.grid()
plt.legend()
plt.title('Car velocity')

plt.show()

