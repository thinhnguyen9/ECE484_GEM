import numpy as np
from math import sin, cos, tan, sqrt, atan2
import matplotlib.pyplot as plt

class CarModel():

    def __init__(self, carLength, steerSpeed, carAccel, carDecel, carDamp, steerLimits, throttleLimits, brakeLimits, steerRateLimits, throttleRateLimits, brakeRateLimits, approx_steerTau):
        self.L = carLength
        self.deltaROC = steerSpeed  # (rad/s) rate of change for steering angle
        self.vAccel = carAccel      # (m/s^2) car acceleration for throttle = 1
        self.vDecel = carDecel      # (m/s^2) car deceleration for brake = 1
        self.vDamp = carDamp        # damping to slow the car down (a = -v_damp*v)
        self.steerTau = approx_steerTau     # approx. time constant (1st-order) of the steering angle (must be slower than actual)
        self.delta_min, self.delta_max = steerLimits[0], steerLimits[1]
        self.deltaROC_min, self.deltaROC_max = steerRateLimits[0], steerRateLimits[1]
        self.throttle_min, self.throttle_max = throttleLimits[0], throttleLimits[1]
        self.throttleROC_min, self.throttleROC_max = throttleRateLimits[0], throttleRateLimits[1]
        self.brake_min, self.brake_max = brakeLimits[0], brakeLimits[1]
        self.brakeROC_min, self.brakeROC_max = brakeRateLimits[0], brakeRateLimits[1]
        self.n = 5  # model dimensions
        self.m = 3
    
    def dx(self, x, u):
        '''
            x = [x; y; theta; delta (steering angle); v (car speed)]
            u = [delta_des; throttle; brake]
        '''
        dx = np.array([ x[4]*cos(x[2]),
                        x[4]*sin(x[2]),
                        x[4]/self.L*tan(x[3]),
                        np.sign(u[0] - x[3])*self.deltaROC,
                        -x[4]*self.vDamp + u[1]*self.vAccel + u[2]*self.vDecel ])
        return dx

    def linearize(self, xs):
        A = np.array([  [0, 0, -xs[4]*sin(xs[2]), 0, cos(xs[2])],
                        [0, 0, xs[4]*cos(xs[2]), 0, sin(xs[2])],
                        [0, 0, 0, xs[4]/(self.L*cos(xs[3])**2), tan(xs[3])/self.L],
                        [0, 0, 0, -1/self.steerTau, 0],
                        [0, 0, 0, 0, -self.vDamp]    ])
        B = np.array([  [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [1/self.steerTau, 0, 0],
                        [0, self.vAccel, self.vDecel] ])
        return A,B

    def saturateControl(self, u, u0, dt):
        if dt > 0:
            # Limit rate
            u[0] = u0[0] + self.saturate((u[0] - u0[0])/dt, self.deltaROC_min, self.deltaROC_max)*dt
            u[1] = u0[1] + self.saturate((u[1] - u0[1])/dt, self.throttleROC_min, self.throttleROC_max)*dt
            u[2] = u0[2] + self.saturate((u[2] - u0[2])/dt, self.brakeROC_min, self.brakeROC_max)*dt
            # Limit value
            u[0] = self.saturate(u[0], self.delta_min, self.delta_max)
            u[1] = self.saturate(u[1], self.throttle_min, self.throttle_max)
            u[2] = self.saturate(u[2], self.brake_min, self.brake_max)
            return u
        else:
            return u0

    def saturate(self, val, lower_bound, upper_bound):
        return max(lower_bound, min(upper_bound, val))
    
    def delta2steer(self, delta):     # delta (rad) to steering wheel angle (rad)
        delta = self.saturate(delta, self.delta_min, self.delta_max)
        return np.sign(delta) * (-6.2109*delta**2 + 21.775*abs(delta))
    
    def steer2delta(self, steeringWheel):     # steering wheel angle (rad) to delta (rad)
        # sgn(delta) = sgn(steeringWheel)
        steeringWheel = self.saturate(steeringWheel, -10.9840, 10.9840)
        if steeringWheel == 0:
            delta = 0.0
        else:
            a = -np.sign(steeringWheel)*6.2109
            b = np.sign(steeringWheel)*21.775
            c = -steeringWheel
            d = (b**2 - 4*a*c) ** .5
            abs_delta = [(-b+d)/(2*a), (-b-d)/(2*a)]
            if abs_delta[0] >= 0 and abs_delta[0] < 0.7:
                delta = abs_delta[0]
            elif abs_delta[1] >= 0 and abs_delta[1] < 0.7:
                delta = abs_delta[1]
            else:
                delta = 0.0
            delta *= np.sign(steeringWheel)
        return delta


class LQR():    # continuous-time only

    def __init__(self, n, m):
        self.type = type
        self.n = n
        self.m = m
    
    def setWeight(self, Q, R):
        self.Q = Q
        self.R = R
        self.R_inv = np.linalg.inv(R)

    def setModel(self, A, B):
        self.A = A
        self.B = B

    def calculateGain(self):
        H = np.vstack((np.hstack((self.A, -self.B.dot(self.R_inv.dot(self.B.T)))),
                        np.hstack((-self.Q, -self.A.T))))
        eigVal, eigVec = np.linalg.eig(H)
        U = eigVec[:, np.argsort(eigVal)]
        U = U[:, :self.n]       # 'stable' subspace - n smallest eigenvalues
        U11 = U[:self.n, :]
        U21 = U[self.n:, :]
        try:
            P = U21.dot(np.linalg.inv(U11))
        except:
            # raise Exception('U11 is not invertible!')
            P = U21.dot(np.linalg.pinv(U11))
        K = self.R_inv.dot(self.B.T.dot(P.real))
        self.K = K
        # return K


class Aux():

    def __init__(self):
        pass

    def point_point_distance(self, p1, p2):
        return np.linalg.norm((p1[0]-p2[0], p1[1]-p2[1]))

    def point_line_distance(self, p, s):  # perpendicular distance from point to line
        """Perpendicular distance from point to line.

            Args:
                p: A tuple (x, y) of the coordinates of the point.
                s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating 2 points on the line.

            Return:
                Euclidean distance from the point to the line.
        """
        # AB = self.vector(s[0], s[1])
        # AE = self.vector(s[0], p)
        AB = s[1] - s[0]
        AE = p - s[0]
        # return abs(self.cross(AB,AE)) / self.norm(AB)
        return np.cross(AB,AE) / np.linalg.norm(AB)
        
    def fit_line(self, p):
        """Least square fit line y = a0 + a1*x.

            Args:
                p = [[x1,y1], [x2,y2], ...]

            Return:
                2 points on the fitted line.
        """
        n = len(p)
        x = [p[i][0] for i in range(n)]
        y = [p[i][1] for i in range(n)]
        a0 = (sum(y)*sum(np.power(x,2)) - sum(x)*sum(np.multiply(x,y))) / (n*sum(np.power(x,2)) - pow(sum(x),2))
        a1 = (n*sum(np.multiply(x,y)) - sum(x)*sum(y)) / (n*sum(np.power(x,2)) - pow(sum(x),2))
        # return [[p[i][0], a0 + a1*p[i][0]] for i in range(2)]

        # tránh trường hợp nhiều điểm trùng x coordinate
        # points = [(p[0][0], a0 + a1*p[0][0])]
        # for point in p:
        #     if point[0] != p[0][0]:
        #         points.append((point[0], a0 + a1*point[0]))
        #         return points
        
        # tránh trường hợp nhiều điểm trùng x coordinate và a1 -> inf
        # nhưng kq có thể ko đúng chiều chuyển động của xe (+/- k*pi)
        ds = 0.1
        dx = ds/sqrt(a1**2 + 1)
        dy = a1*ds/sqrt(a1**2 + 1)
        return [(p[0][0], a0 + a1*p[0][0]),
                (p[0][0] + dx, a0 + a1*p[0][0] + dy)]
    
    def ErrorsFromWaypoints(self, currState, wpList):
        """
            Args:
                currState = [x,y,theta]
                wpList = [[x1,y1], [x2,y2], ...]

            Return:
                Cross-track err: perp. distance to the fitted line of wpList.
                Heading err: angle difference to the fitted line of wpList.
        """
        # currLoc = currState[:2]
        # currTheta = currState[2]
        # wp = np.array(self.fit_line(wpList))    # np.array([[x1,y1],[x2,y2]])
        # ct_err = self.point_line_distance(currLoc, wp)
        # path = wp[1] - wp[0]
        # hd_err = currTheta - atan2(path[1], path[0])
        # while hd_err > np.pi:
        #     hd_err = hd_err - 2*np.pi
        # while hd_err < -np.pi:
        #     hd_err = hd_err + 2*np.pi
        # return ct_err, hd_err
    
        # tránh trường hợp wpList ko đúng chiều di chuyển của xe
        currLoc = np.array(currState[:2])
        currTheta = np.array(currState[2])
        wp = np.array(self.fit_line(wpList))    # np.array([[x1,y1],[x2,y2]])
        path = wp[1] - wp[0]
        hd_err = currTheta - atan2(path[1], path[0])
        while hd_err > np.pi:
            hd_err = hd_err - 2*np.pi
        while hd_err < -np.pi:
            hd_err = hd_err + 2*np.pi
        
        if hd_err > np.pi/2 or hd_err < -np.pi/2:
            wp = [wp[1], wp[0]]     # đảo thứ tự
            path = wp[1] - wp[0]
            hd_err = currTheta - atan2(path[1], path[0])
            while hd_err > np.pi:
                hd_err = hd_err - 2*np.pi
            while hd_err < -np.pi:
                hd_err = hd_err + 2*np.pi
        ct_err = self.point_line_distance(currLoc, wp)
        # print(np.int_(currLoc*100))
        # print(np.int_(wp*100))
        # print("\n")
        return ct_err, hd_err

    def ll2xy(self, lat, lon, olat, olon):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        # lon_wp_x, lat_wp_y = alvinxy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        # https://docs.ros.org/en/jade/api/geonav_transform/html/_modules/alvinxy/alvinxy.html#ll2xy

        latrad = olat*np.pi/180.0
        mdeglon = 111415.13*cos(latrad) - 94.55*cos(3.0*latrad) + 0.12*cos(5.0*latrad)
        mdeglat = 111132.09 - 566.05*cos(2.0*latrad) + 1.20*cos(4.0*latrad) - 0.002*cos(6.0*latrad)

        x = (lon - olon)*mdeglon
        y = (lat - olat)*mdeglat
        return x, y

